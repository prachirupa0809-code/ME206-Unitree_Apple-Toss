import os
import time
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer

# TUNABLE SETTINGS
T_FLIGHT = 0.42          # Time (seconds) the apple spends in the air
SCOOP_DIP = 0.065        # How far down (m) the hand dips before throwing
SCOOP_RISE = 0.075       # How far up (m) the hand rises to release
CATCH_CUSHION = 0.045    # Downward give (m) when catching the apple

HOLD_KP = 220.0          # Virtual spring stiffness
HOLD_KD = 24.0           # Overdamped velocity damping (zero bounce)

# Ready palm-center targets relative to pelvis [x (forward), y (left/right), z (up)]
LEFT_READY_OFFSET = np.array([0.27, 0.17, 0.06])
RIGHT_READY_OFFSET = np.array([0.27, -0.17, 0.06])

# XML GENERATION (g1_fixed.xml & g1_toss.xml)
def prepare_xmls(menagerie_dir: str) -> str:
    g1_dir = os.path.join(menagerie_dir, "unitree_g1")
    base_xml = os.path.join(g1_dir, "g1_with_hands.xml")
    if not os.path.exists(base_xml):
        base_xml = os.path.join(g1_dir, "g1.xml")
    if not os.path.exists(base_xml):
        raise FileNotFoundError(f"Could not find g1.xml in {g1_dir}")

    fixed_xml = os.path.join(g1_dir, "g1_fixed.xml")
    toss_xml = os.path.join(g1_dir, "g1_toss.xml")

    tree = ET.parse(base_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Fix pelvis upright in space
    pelvis = None
    for body in worldbody.findall("body"):
        if "pelvis" in body.attrib.get("name", ""):
            pelvis = body
            break

    if pelvis is not None:
        for jnt in list(pelvis.findall("freejoint")) + list(pelvis.findall("joint")):
            if jnt.tag == "freejoint" or jnt.attrib.get("type") == "free":
                pelvis.remove(jnt)
        pelvis.set("pos", "0 0 0.80")

    # Keep critically damped contact on hands/fingers; disable on torso/arms
    hand_keywords = ["wrist", "hand", "palm", "thumb", "index", "middle", "ring", "pinky"]
    for b in root.iter("body"):
        bname = b.attrib.get("name", "").lower()
        is_hand = any(k in bname for k in hand_keywords)
        for geom in b.findall("geom"):
            if is_hand:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("solref", "0.025 1.5")
                geom.set("friction", "1.2 0.01 0.001")
            else:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    # Inject a palm_center <site> inside each hand/wrist body
    for side in ["left", "right"]:
        sign = -1.0 if side == "left" else 1.0
        target_body = None
        for b in root.iter("body"):
            bname = b.attrib.get("name", "")
            if side in bname and ("wrist_yaw" in bname or "palm" in bname or "rubber_hand" in bname):
                target_body = b
        if target_body is not None:
            ET.SubElement(target_body, "site", {
                "name": f"{side}_palm_site",
                "pos": f"0.095 {0.032 * sign:.4f} 0.0",
                "size": "0.008",
                "rgba": "0 1 0 0.0"
            })

    tree.write(fixed_xml, encoding="unicode")

    # Build g1_toss.xml
    toss_root = ET.Element("mujoco", {"model": "g1_apple_toss"})
    ET.SubElement(toss_root, "include", {"file": "g1_fixed.xml"})
    ET.SubElement(toss_root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})

    visual = ET.SubElement(toss_root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(visual, "headlight", {"diffuse": "0.75 0.75 0.75", "ambient": "0.3 0.3 0.3"})

    asset = ET.SubElement(toss_root, "asset")
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "rgb1": "0.86 0.88 0.91", "rgb2": "0.76 0.79 0.83",
        "width": "512", "height": "512"
    })
    ET.SubElement(asset, "material", {
        "name": "grid_mat", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.1"
    })

    wb = ET.SubElement(toss_root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "1.5 0 2.5", "dir": "-0.5 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane", "size": "5 5 0.1",
        "material": "grid_mat", "contype": "1", "conaffinity": "1"
    })

    apple = ET.SubElement(wb, "body", {"name": "apple", "pos": "0.27 0.17 0.90"})
    ET.SubElement(apple, "freejoint", {"name": "apple_joint"})
    ET.SubElement(apple, "geom", {
        "name": "apple_geom", "type": "sphere", "size": "0.036",
        "mass": "0.15", "rgba": "0.88 0.12 0.12 1",
        "contype": "1", "conaffinity": "1",
        "solref": "0.025 1.5",
        "friction": "1.2 0.01 0.001"
    })
    ET.SubElement(apple, "geom", {
        "name": "apple_stem", "type": "cylinder", "size": "0.004 0.012",
        "pos": "0 0 0.040", "rgba": "0.35 0.2 0.05 1",
        "contype": "0", "conaffinity": "0"
    })

    ET.ElementTree(toss_root).write(toss_xml, encoding="unicode")
    return toss_xml

# 6-DOF ARM IK (POSITION + PALM-UP ORIENTATION) & FINGER CONTROLLER
class ArmAndHandController:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.ik_data = mujoco.MjData(model)

        self.left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm_site")
        self.right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site")

        self.left_arm_dofs = self._find_arm_joints("left")
        self.right_arm_dofs = self._find_arm_joints("right")

        self.left_fingers = self._find_finger_actuators("left")
        self.right_fingers = self._find_finger_actuators("right")

    def _find_arm_joints(self, side: str):
        keywords = ["shoulder", "elbow", "wrist"]
        dofs = []
        for i in range(self.model.nu):
            j_id = self.model.actuator_trnid[i, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or ""
            if side in jname and any(k in jname for k in keywords):
                dofs.append((j_id, self.model.jnt_qposadr[j_id], self.model.jnt_dofadr[j_id], i))
        return dofs

    def _find_finger_actuators(self, side: str):
        finger_keywords = ["thumb", "index", "middle", "ring", "pinky", "hand"]
        acts = []
        for i in range(self.model.nu):
            j_id = self.model.actuator_trnid[i, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or ""
            if side in jname and any(k in jname for k in finger_keywords):
                lo, hi = self.model.actuator_ctrlrange[i]
                curl_val = hi if abs(hi) > abs(lo) else lo
                acts.append((i, jname, curl_val))
        return acts

    def get_finger_ctrl(self, side: str, curl_ratio: float) -> dict:
        acts = self.left_fingers if side == "left" else self.right_fingers
        out = {}
        for act_id, jname, curl_val in acts:
            if "thumb_0" in jname:
                scale = 0.30
            elif "thumb" in jname:
                scale = 0.36
            else:
                scale = 0.42
            out[act_id] = curl_ratio * scale * curl_val
        return out

    def _palm_orientation_error(self, site_id: int, side: str) -> np.ndarray:
        R = self.ik_data.site_xmat[site_id].reshape(3, 3)
        palm_normal = -R[:, 1] if side == "left" else R[:, 1]
        desired_normal = np.array([0.0, 0.0, 1.0])

        finger_dir = R[:, 0]
        desired_finger = np.array([0.95, -0.31 if side == "left" else 0.31, 0.0])
        desired_finger /= np.linalg.norm(desired_finger)

        return np.cross(palm_normal, desired_normal) + 0.5 * np.cross(finger_dir, desired_finger)

    def solve_ik(self, current_qpos: np.ndarray, left_pos: np.ndarray, right_pos: np.ndarray, max_iters: int = 12):
        self.ik_data.qpos[:] = current_qpos[:]
        damping = 1.5e-2

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.ik_data)

            l_err_p = left_pos - self.ik_data.site_xpos[self.left_site]
            l_err_r = self._palm_orientation_error(self.left_site, "left")
            l_err = np.concatenate([l_err_p, 0.35 * l_err_r])

            jacp_l = np.zeros((3, self.model.nv))
            jacr_l = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.ik_data, jacp_l, jacr_l, self.left_site)
            l_cols = [d[2] for d in self.left_arm_dofs]
            Jl = np.vstack([jacp_l[:, l_cols], 0.35 * jacr_l[:, l_cols]])
            dq_l = Jl.T @ np.linalg.solve(Jl @ Jl.T + damping * np.eye(6), l_err)

            r_err_p = right_pos - self.ik_data.site_xpos[self.right_site]
            r_err_r = self._palm_orientation_error(self.right_site, "right")
            r_err = np.concatenate([r_err_p, 0.35 * r_err_r])

            jacp_r = np.zeros((3, self.model.nv))
            jacr_r = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.ik_data, jacp_r, jacr_r, self.right_site)
            r_cols = [d[2] for d in self.right_arm_dofs]
            Jr = np.vstack([jacp_r[:, r_cols], 0.35 * jacr_r[:, r_cols]])
            dq_r = Jr.T @ np.linalg.solve(Jr @ Jr.T + damping * np.eye(6), r_err)

            if np.linalg.norm(l_err_p) < 2e-3 and np.linalg.norm(r_err_p) < 2e-3 and np.linalg.norm(l_err_r) < 0.05:
                break

            for idx, (j_id, qpos_adr, _, _) in enumerate(self.left_arm_dofs):
                lo, hi = self.model.jnt_range[j_id]
                self.ik_data.qpos[qpos_adr] = np.clip(self.ik_data.qpos[qpos_adr] + 0.65 * dq_l[idx], lo, hi)

            for idx, (j_id, qpos_adr, _, _) in enumerate(self.right_arm_dofs):
                lo, hi = self.model.jnt_range[j_id]
                self.ik_data.qpos[qpos_adr] = np.clip(self.ik_data.qpos[qpos_adr] + 0.65 * dq_r[idx], lo, hi)

        mujoco.mj_forward(self.model, self.ik_data)
        ctrl = np.zeros(self.model.nu)
        for j_id, qpos_adr, _, act_id in self.left_arm_dofs + self.right_arm_dofs:
            ctrl[act_id] = self.ik_data.qpos[qpos_adr]
        return ctrl

# SIMULATION LOOP
def run_simulation(args):
    toss_xml = prepare_xmls(args.menagerie)
    model = mujoco.MjModel.from_xml_path(toss_xml)
    data = mujoco.MjData(model)
    ctrl_helper = ArmAndHandController(model)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    apple_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    apple_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "apple_joint")
    apple_qpos = model.jnt_qposadr[apple_jnt_id]
    apple_dof = model.jnt_dofadr[apple_jnt_id]

    mujoco.mj_forward(model, data)
    base_pos = data.xpos[pelvis_id].copy()
    left_ready = base_pos + LEFT_READY_OFFSET
    right_ready = base_pos + RIGHT_READY_OFFSET

    # Initialize arms into palm-up ready pose with gentle cradle curl
    init_ctrl = ctrl_helper.solve_ik(data.qpos, left_ready, right_ready, max_iters=40)
    for act_id, val in ctrl_helper.get_finger_ctrl("left", 0.85).items():
        init_ctrl[act_id] = val
    for act_id, val in ctrl_helper.get_finger_ctrl("right", 0.0).items():
        init_ctrl[act_id] = val

    for j_id, qpos_adr, _, act_id in ctrl_helper.left_arm_dofs + ctrl_helper.right_arm_dofs:
        data.qpos[qpos_adr] = init_ctrl[act_id]
    data.ctrl[:] = init_ctrl
    mujoco.mj_forward(model, data)

    apple_palm_offset = np.array([0.0, 0.0, 0.046])
    data.qpos[apple_qpos : apple_qpos + 3] = data.site_xpos[ctrl_helper.left_site] + apple_palm_offset
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    fps = 30
    frame_interval = 1.0 / fps
    next_frame_time = 0.0
    if args.record:
        renderer = mujoco.Renderer(model, height=720, width=1280)

    # Snappy, natural juggling rhythm (no long idle gap between tosses)
    t_hold = 0.16
    t_dip = 0.24
    t_scoop = 0.18
    t_air = T_FLIGHT
    t_cushion = 0.22
    t_settle = 0.14
    toss_period = t_hold + t_dip + t_scoop + t_air + t_cushion + t_settle

    toss_count = 0
    local_t = 0.0
    throwing_side = "left"
    released = False
    cached_arm_ctrl = init_ctrl.copy()
    substep_counter = 0

    prev_site_pos = {
        ctrl_helper.left_site: data.site_xpos[ctrl_helper.left_site].copy(),
        ctrl_helper.right_site: data.site_xpos[ctrl_helper.right_site].copy(),
    }

    def step_logic(dt):
        nonlocal local_t, toss_count, throwing_side, released, cached_arm_ctrl, substep_counter

        local_t += dt
        if local_t >= toss_period:
            local_t -= toss_period
            toss_count += 1
            throwing_side = "right" if throwing_side == "left" else "left"
            released = False

        throw_ready = left_ready if throwing_side == "left" else right_ready
        catch_ready = right_ready if throwing_side == "left" else left_ready
        throw_site = ctrl_helper.left_site if throwing_side == "left" else ctrl_helper.right_site
        catch_site = ctrl_helper.right_site if throwing_side == "left" else ctrl_helper.left_site

        throw_target = throw_ready.copy()
        catch_target = catch_ready.copy()
        active_site = None

        throw_curl = 0.0
        catch_curl = 0.0

        if local_t < t_hold:
            active_site = throw_site
            throw_curl = 0.85
            released = False

        elif local_t < t_hold + t_dip:
            u = (local_t - t_hold) / t_dip
            s = 0.5 - 0.5 * np.cos(np.pi * u)
            throw_target[2] -= SCOOP_DIP * s
            throw_curl = 0.85 * (1.0 - s)
            active_site = throw_site

        elif local_t < t_hold + t_dip + t_scoop:
            u = (local_t - (t_hold + t_dip)) / t_scoop
            s = 0.5 - 0.5 * np.cos(np.pi * u)
            throw_target[2] += -SCOOP_DIP + (SCOOP_DIP + SCOOP_RISE) * s
            inward = 0.035 * s
            throw_target[1] += -inward if throwing_side == "left" else inward
            throw_curl = 0.0
            active_site = throw_site

        elif local_t < t_hold + t_dip + t_scoop + t_air:
            if not released:
                released = True
                p0 = data.qpos[apple_qpos : apple_qpos + 3].copy()
                p1 = catch_ready + apple_palm_offset
                g = 9.81
                vx = (p1[0] - p0[0]) / T_FLIGHT
                vy = (p1[1] - p0[1]) / T_FLIGHT
                vz = (p1[2] - p0[2] + 0.5 * g * T_FLIGHT**2) / T_FLIGHT
                data.qvel[apple_dof : apple_dof + 3] = [vx, vy, vz]
                data.qvel[apple_dof + 3 : apple_dof + 6] = [0.0, 1.5, 0.0]

            u = (local_t - (t_hold + t_dip + t_scoop)) / t_air
            throw_target[2] += SCOOP_RISE * (1.0 - min(u * 2.0, 1.0))
            throw_curl = 0.0
            catch_curl = 0.0
            active_site = None

        elif local_t < t_hold + t_dip + t_scoop + t_air + t_cushion:
            u = (local_t - (t_hold + t_dip + t_scoop + t_air)) / t_cushion
            cushion = CATCH_CUSHION * np.sin(np.pi * u)
            catch_target[2] -= cushion
            curl_progress = max(0.0, (u - 0.25) / 0.75)
            catch_curl = 0.85 * curl_progress
            active_site = catch_site

        else:
            catch_curl = 0.85
            active_site = catch_site

        # Solve IK every 4 physics steps (125 Hz) for fast real-time speed
        if substep_counter % 4 == 0:
            l_tgt = throw_target if throwing_side == "left" else catch_target
            r_tgt = catch_target if throwing_side == "left" else throw_target
            cached_arm_ctrl = ctrl_helper.solve_ik(data.qpos, l_tgt, r_tgt, max_iters=10)
        substep_counter += 1

        ctrl = cached_arm_ctrl.copy()
        l_curl = throw_curl if throwing_side == "left" else catch_curl
        r_curl = catch_curl if throwing_side == "left" else throw_curl
        for act_id, val in ctrl_helper.get_finger_ctrl("left", l_curl).items():
            ctrl[act_id] = val
        for act_id, val in ctrl_helper.get_finger_ctrl("right", r_curl).items():
            ctrl[act_id] = val

        data.ctrl[:] = ctrl

        # Overdamped velocity-tracking spring-damper
        data.xfrc_applied[apple_body_id, :] = 0.0
        if active_site is not None:
            curr_palm_pos = data.site_xpos[active_site].copy()
            palm_vel = (curr_palm_pos - prev_site_pos[active_site]) / dt
            target_apple_pos = curr_palm_pos + apple_palm_offset

            apple_pos = data.qpos[apple_qpos : apple_qpos + 3]
            apple_vel = data.qvel[apple_dof : apple_dof + 3]

            f_spring = HOLD_KP * (target_apple_pos - apple_pos) + HOLD_KD * (palm_vel - apple_vel)
            f_grav = np.array([0.0, 0.0, 0.15 * 9.81])
            data.xfrc_applied[apple_body_id, :3] = f_spring + f_grav
            data.qvel[apple_dof + 3 : apple_dof + 6] *= 0.85

        prev_site_pos[ctrl_helper.left_site] = data.site_xpos[ctrl_helper.left_site].copy()
        prev_site_pos[ctrl_helper.right_site] = data.site_xpos[ctrl_helper.right_site].copy()

    if args.record:
        print(f"Recording {args.tosses} tosses to {args.record}...")
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.24, 0.0, 0.90]
        cam.distance = 1.25
        cam.azimuth = 145
        cam.elevation = -18

        while toss_count < args.tosses:
            step_logic(model.opt.timestep)
            mujoco.mj_step(model, data)
            if data.time >= next_frame_time:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
                next_frame_time += frame_interval

        import imageio.v2 as imageio
        imageio.mimsave(args.record, frames, fps=fps)
        print(f"Saved {args.record} ({len(frames)} frames).")
    else:
        print("Launching interactive viewer at 1.0x real-time speed... Close window or press Ctrl+C to stop.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.24, 0.0, 0.90]
            viewer.cam.distance = 1.25
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -18

            render_fps = 60.0
            render_dt = 1.0 / render_fps
            steps_per_frame = max(1, int(round(render_dt / model.opt.timestep)))

            while viewer.is_running():
                if not args.loop and toss_count >= args.tosses:
                    break

                frame_start = time.perf_counter()
                for _ in range(steps_per_frame):
                    step_logic(model.opt.timestep)
                    mujoco.mj_step(model, data)

                viewer.sync()

                # Sleep once per 60 FPS visual frame instead of per 2ms physics step
                elapsed = time.perf_counter() - frame_start
                remaining = render_dt - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree G1 Apple Toss")
    parser.add_argument("--menagerie", type=str, default=r"C:\Users\Prerna\mujoco_menagerie")
    parser.add_argument("--tosses", type=int, default=6)
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--record", type=str, default="")
    args = parser.parse_args()
    run_simulation(args)