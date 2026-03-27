from pathlib import Path
from re import escape
from typing import Dict, List, Tuple, Literal
from jaxtyping import Float
from torch import Tensor
import torch
import torchmetrics
import numpy as np
import math
from omegaconf import OmegaConf
from argparse import ArgumentParser
from tqdm import tqdm
from omegaconf import OmegaConf

from libs.utils.fncsmpl import SmplModel
from libs.inference.dfot_inference import load_inf_cfg
from omegaconf import OmegaConf

torch.set_printoptions(precision=3, sci_mode=False)

class EvalMotion:
    def __init__(self) -> None:
        # self.smpl_model = SmplModel.load(smpl_model_path)
        self.random_seed = 1234
        self.fps = 30

    def eval_from_dir(
        self,
        data_dir: Path,
        fps: int = 30,
        sample_num: int | None = None,
        is_whole_seq: bool = False,
        seq_len: int | None = None,
        test_set_path: Path | None = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.fps = fps
        assert data_dir.exists()
        if test_set_path is None:
            raw_data = [np.load(p, allow_pickle=True) for p in data_dir.glob("*.npz")]
        else:
            if not test_set_path.exists():
                raise ValueError("Test set path does not exist.")
            test_set_dict = OmegaConf.load(test_set_path)
            raw_data = [np.load(data_dir/f"{p}.npz", allow_pickle=True) for p in test_set_dict['test']]
        if raw_data == []:
            raise ValueError("No data found in the data directory.")

        # config = load_inf_cfg(data_dir/'inf_cfg.yaml')
        config = OmegaConf.load(data_dir/'inf_cfg.yaml')

        sampling_cfg = config.get("sampling_cfg", None)
        if sampling_cfg is not None:
            task_mode = sampling_cfg.get("task_mode", "joint")
            print(f'mode: {task_mode}')
            print(f'context: {sampling_cfg.context_seq_len}')
        else:
            task_mode = "joint"
            print("mode: vqvae (no sampling_cfg)")
        print(f'sample_num: {sample_num}')

        is_eval_last_person = task_mode in ["partner_prediction", "motion_control", "partner_inpainting"]
        # data = [{k: torch.from_numpy(v).to(device) for k, v in d.items()} for d in raw_data]

        data = []
        for d in raw_data:
            data.append({})
            for k, v in d.items():
                if isinstance(v, np.ndarray) and v.dtype.kind in ('f', 'i', 'u', 'b'):
                    data[-1][k] = torch.from_numpy(v).to(device)
        
        eval_dict = self.eval(
            data, 
            diversity_times=300, 
            smpl_path=config.smpl_path, 
            is_eval_last_person=is_eval_last_person,
            sample_num=sample_num,
            is_whole_seq=is_whole_seq,
            seq_len=seq_len,
            device=device)

        suffix = f"_seq_{seq_len}" if seq_len is not None else ""
        suffix = "_whole_seq" if is_whole_seq else suffix
        suffix = "_test_set" if test_set_path is not None else suffix
        OmegaConf.save(OmegaConf.create(eval_dict), data_dir/f'eval_dict_c{suffix}_1113.yaml')
        print(f"Saved {data_dir/f'eval_dict_c{suffix}_1113.yaml'}")

        csv_lines = self.convert_to_csv(eval_dict)
        with open(data_dir/f'eval_dict_c{suffix}_1113.csv', 'w') as f:
            for line in csv_lines:
                f.write(line + "\n")
        print(f"Saved {data_dir/f'eval_dict_c{suffix}_1113.csv'}")

    def eval_from_dir_val(
        self,
        data_dir: Path,
        sample_num: int | None = None,
        is_whole_seq: bool = False,
        seq_len: int | None = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        assert data_dir.exists()
        raw_data = [np.load(p, allow_pickle=True) for p in data_dir.glob("*.npz")]
        if raw_data == []:
            raise ValueError("No data found in the data directory.")
        config = load_inf_cfg(data_dir/'inf_cfg.yaml')
        print(f'mode: {config.sampling_cfg.task_mode}')
        print(f'context: {config.sampling_cfg.context_seq_len}')
        print(f'sample_num: {sample_num}')

        is_eval_last_person = config.sampling_cfg.task_mode in ["partner_prediction", "motion_control"]
        data = [{k: torch.from_numpy(v).to(device) for k, v in d.items()} for d in raw_data]

        smpl_path = Path("./body_model/smplx/SMPLX_NEUTRAL.npz")
        self.smpl_model = SmplModel.load(smpl_path).to(device)
        gt_feature_dict, gen_feature_dict, person_num = self.calc_feature_dict(
            data, sample_num=sample_num, is_whole_seq=is_whole_seq, seq_len=seq_len) 

        return gt_feature_dict, gen_feature_dict  

    def eval(
        self,
        motion_list: List[Dict[str, Float[Tensor, "B T P ..."]]],
        diversity_times: int = 300,
        smpl_path: Path = Path("./body_model/smplx/SMPLX_NEUTRAL.npz"),
        is_eval_last_person: bool = False,
        sample_num: int | None = None,
        is_whole_seq: bool = False,
        seq_len: int | None = None,
        device: torch.device = torch.device("cuda"),
    ) -> Dict[str, float]:
        self.smpl_model = SmplModel.load(smpl_path).to(device)
        gt_feature_dict, gen_feature_dict, person_num = self.calc_feature_dict(
            motion_list, sample_num=sample_num, is_whole_seq=is_whole_seq, seq_len=seq_len)
        eval_dict = {}

        generator = torch.Generator(device=device).manual_seed(self.random_seed)

        # for key0 in ["body"]:
        #     for key1 in ["g", "k"]:
        #         print(key0, key1, 'gt', gt_feature_dict[key0][key1][0][0, 0, -1:, :])
        #         print(key0, key1, 'gen', gen_feature_dict[key0][key1][0][0, 0, -1:, :])
        #         diff = torch.linalg.vector_norm(gen_feature_dict[key0][key1][0][0, 0, -1:, :] - gt_feature_dict[key0][key1][0][0, 0, -1:, :])
        #         print(diff)                
            

        for key0 in ["body"]:
            eval_dict[key0] = {"g": {}, "k": {}} #, "g_woc": {}, "k_woc": {}}
            for key1 in ["g", "k"]: #, "g_woc", "k_woc"]:
                for mode in ["solo"]: #, "multi"]:
                    if mode == "multi" and (person_num == 1 or is_eval_last_person):
                        continue
                    if is_eval_last_person:
                        gt_feature_list = [v[:, :, -1:] for v in gt_feature_dict[key0][key1]]
                        gen_feature_list = [v[:, :, -1:] for v in gen_feature_dict[key0][key1]]
                    else:
                        gt_feature_list = gt_feature_dict[key0][key1]
                        gen_feature_list = gen_feature_dict[key0][key1]
                    gt_feature, gt_mu, gt_sigma = self.calc_mu_sigma(
                        gt_feature_list, mode=mode)
                    gen_feature, gen_mu, gen_sigma = self.calc_mu_sigma(
                        gen_feature_list, mode=mode)
                    # print(gt_sigma, gen_sigma)
                    fd = self.calc_frechet_distance(gt_mu, gt_sigma, gen_mu, gen_sigma)
                    # div_gt = self.calc_diversity_new(
                    #     gt_feature_list, is_seq=False)
                    # div_gen = self.calc_diversity_new(
                    #     gen_feature_list, is_seq=False)
                    div_sample = self.calc_sample_diversity_new(
                        gen_feature_list, is_seq=False)
                    print(f"{key0} {key1} {mode} fd: {fd:.4f}, div_sample: {div_sample:.4f}")
                    seq = 30
                    stride = 10
                    gt_feature_seq, gt_mu_seq, gt_sigma_seq = self.calc_mu_sigma_seq(
                        gt_feature_list, seq=seq, stride=stride, mode=mode)
                    gen_feature_seq, gen_mu_seq, gen_sigma_seq = self.calc_mu_sigma_seq(
                        gen_feature_list, seq=seq, stride=stride, mode=mode)
                    fd_seq = self.calc_frechet_distance(gt_mu_seq, gt_sigma_seq, gen_mu_seq, gen_sigma_seq) / seq
                    # div_gt_seq = self.calc_diversity_new(
                    #     gt_feature_list, is_seq=True)
                    # div_gen_seq = self.calc_diversity_new(
                    #     gen_feature_list, is_seq=True)
                    div_sample_seq = self.calc_sample_diversity_new(
                        gen_feature_list, is_seq=True)
                    print(f"{key0} {key1} {mode} fd_seq: {fd_seq:.4f}, div_sample_seq: {div_sample_seq:.4f}")
                    eval_dict[key0][key1][mode] = {
                        "fd": fd,
                        "div_sample": div_sample,
                    }
                    eval_dict[key0][key1][mode+"_seq"] = {
                        "fd": fd_seq,
                        "div_sample": div_sample_seq,
                    }

        for key0 in ["body"]:
            for key1 in ["g", "k"]: #, "g_woc", "k_woc"]:
                print(key0, key1, gen_feature_dict[key0][key1][0][0, 0, -1:, :20])


        mpje_key_dict = {"g": "mpjpe", "k": "mpjve", "g_wor": "pa_mpjpe", "k_wor": "pa_mpjve"}
        for key0 in ["body"]:
            for key1 in ["g", "k"]: #, "g_wor", "k_wor"]:
                mpje_key = mpje_key_dict[key1]
                if is_eval_last_person:
                    gt_feature = [v[:, :, -1:] for v in gt_feature_dict[key0][key1]]
                    gen_feature = [v[:, :, -1:] for v in gen_feature_dict[key0][key1]]
                else:
                    gt_feature = gt_feature_dict[key0][key1]
                    gen_feature = gen_feature_dict[key0][key1]

                # print(key0, key1, gen_feature_dict[key0][key1][0][0, 0, -1:, :20])
                # print(key0, key1, gt_feature_dict[key0][key1][0][0, 0, -1:, :20])
                # print(key0, key1, gen_feature[0][0, 0, 0, :20])
                # print(key0, key1, gt_feature[0][0, 0, 0, :20])
                # diff = torch.linalg.vector_norm(gen_feature_dict[key0][key1][0][0, 0, 0, :] - gt_feature_dict[key0][key1][0][0, 0, 0, :])
                # print(key0, key1, diff)
                
                mpje, best_mpje = self.calc_mean_per_joint_error(
                    gt_feature, gen_feature)
                eval_dict[key0][mpje_key] = mpje
                eval_dict[key0][mpje_key+"_best_list"] = best_mpje
                f_best_mpje = ', '.join([f"{p:.3f}" for p in best_mpje])
                print(f"{key0} {mpje_key}: {mpje:.4f} | [{f_best_mpje}]")

        for key0 in ["body"]:
            for key1 in ["g", "k"]: #, "g_woc", "k_woc"]:
                if person_num == 1:
                    continue
                person_corr_gt = self.calc_person_correlation(
                    gt_feature_dict[key0][key1],
                )
                person_corr_gen = self.calc_person_correlation(
                    gen_feature_dict[key0][key1],
                )
                print(f"{key0} {key1} corr_gt: {person_corr_gt:.4f}, corr_gen: {person_corr_gen:.4f}")
                eval_dict[key0][key1]["corr_gt"] = person_corr_gt
                eval_dict[key0][key1]["corr_gen"] = person_corr_gen


        if is_eval_last_person:
            gt_feature = [v[:, :, -1:] for v in gt_feature_dict["body"]["g"]]
            gen_feature = [v[:, :, -1:] for v in gen_feature_dict["body"]["g"]]
        else:
            gt_feature = gt_feature_dict["body"]["g"]
            gen_feature = gen_feature_dict["body"]["g"]


        foot_skate_gt = self.calc_foot_skate_metric(gt_feature)
        foot_skate_gen = self.calc_foot_skate_metric(gen_feature)
        print(f"foot_skate_gt: {foot_skate_gt}")
        print(f"foot_skate_gen: {foot_skate_gen}")
        eval_dict["body"]["g"]["foot_skate_gt"] = foot_skate_gt
        eval_dict["body"]["g"]["foot_skate_gen"] = foot_skate_gen


        penetration_gt = self.calc_penetration_metric_fast(
            gt_feature_dict["body"]["g"],
            gt_feature_dict['root_pos'],
        )
        penetration_gen = self.calc_penetration_metric_fast(
            gen_feature_dict["body"]["g"],
            gen_feature_dict['root_pos'],
        )
        print(f"penetration_gt: {penetration_gt}")
        print(f"penetration_gen: {penetration_gen}")
        eval_dict["body"]["g"]['penetration_gt'] = penetration_gt
        eval_dict["body"]["g"]['penetration_gen'] = penetration_gen      

        return eval_dict

    def convert_to_csv(
        self, 
        eval_dict: Dict[str, Dict[str, Dict[str, float]]],
    ):
        gt_dict = {}
        gen_dict = {}
        def get_key(key: str):
            new_key = key.replace("_gt", "")
            is_gt = new_key != key
            new_key = new_key.replace("_gen", "")
            return new_key, is_gt
        def register(key: str, val: float, is_gt: bool):
            if isinstance(val, float):
                if is_gt:
                    gt_dict[key] = val
                    gen_dict[key] = '' if not key in gen_dict.keys() else gen_dict[key]
                else:
                    gen_dict[key] = val
                gt_dict[key] = '' if not key in gt_dict.keys() else gt_dict[key]
            else:
                print(f"skip {key}")
    
        for k0, v0 in eval_dict["body"].items():
            if isinstance(v0, float):
                key, is_gt = get_key(f"{k0}")
                register(key, v0, is_gt)
            if isinstance(v0, dict):
                for k1, v1 in v0.items():
                    if isinstance(v1, float):
                        key, is_gt = get_key(f"{k0}.{k1}")
                        register(key, v1, is_gt)
                    if isinstance(v1, dict):
                        for k2, v2 in v1.items():
                            if isinstance(v2, float):
                                key, is_gt = get_key(f"{k0}.{k1}.{k2}")
                                register(key, v2, is_gt)
        key_list = list(gt_dict.keys())
        val_line_gt = "gt," + ",".join([str(gt_dict[key]) for key in key_list])
        val_line_gen = "gen," + ",".join([str(gen_dict[key]) for key in key_list])
        key_line = "split," + ",".join(key_list)
        lines = [key_line, val_line_gt, val_line_gen]
        return lines


    def calc_feature_dict(
        self,
        motion_list: List[Dict[str, Float[Tensor, "S+1 T P ..."]]],
        sample_num: int | None = None,
        is_whole_seq: bool = False,
        seq_len: int | None = None,
    ) -> Tuple[List[Dict[str, List[Float[Tensor, "1 T P D"]]]], List[Dict[str, List[Float[Tensor, "S T P D"]]]], torch.device]:
        keys0 = ["body", "hand"]
        keys1 = ["g", "k", "g_woc", "k_woc", "g_wor", "k_wor"]
        keys0_ = ["root_pos"]
        gt_feature = {key0: {key1: [] for key1 in keys1} for key0 in keys0}
        gen_feature = {key0: {key1: [] for key1 in keys1} for key0 in keys0}
        gt_feature.update({key0: [] for key0 in keys0_})
        gen_feature.update({key0: [] for key0 in keys0_})
        sample_num_ = motion_list[0]["betas"].shape[0] if sample_num is None else sample_num+1

        for motion in tqdm(motion_list, desc="calc feature"):
            gt_seq_len = motion["gt_timesteps"] if "gt_timesteps" in motion else motion["timesteps"]
            ctx_seq_len = motion["context_timesteps"]
            tqdm.write(f'context: {ctx_seq_len}, gt: {gt_seq_len}')
            
            if is_whole_seq:
                gen_f = self.calc_feature(
                    betas = motion["betas"][1:, ctx_seq_len:],
                    T_world_root = motion["T_world_root"][1:, ctx_seq_len:],
                    body_joint_rotations = motion["body_joint_rotations"][1:, ctx_seq_len:],
                    hand_joint_rotations = motion["hand_joint_rotations"][1:, ctx_seq_len:],
                    # device=torch.device("cpu"),
                )
            elif seq_len is not None:
                gen_f = self.calc_feature(
                    betas = motion["betas"][1:, ctx_seq_len:seq_len],
                    T_world_root = motion["T_world_root"][1:, ctx_seq_len:seq_len],
                    body_joint_rotations = motion["body_joint_rotations"][1:, ctx_seq_len:seq_len],
                    hand_joint_rotations = motion["hand_joint_rotations"][1:, ctx_seq_len:seq_len],
                    # device=torch.device("cpu"),
                )
                gt_seq_len = min(gt_seq_len, seq_len)
            else:
                gen_f = self.calc_feature(
                    betas = motion["betas"][1:sample_num_, ctx_seq_len:gt_seq_len],
                    T_world_root = motion["T_world_root"][1:sample_num_, ctx_seq_len:gt_seq_len],
                    body_joint_rotations = motion["body_joint_rotations"][1:sample_num_, ctx_seq_len:gt_seq_len],
                    hand_joint_rotations = motion["hand_joint_rotations"][1:sample_num_, ctx_seq_len:gt_seq_len],
                    # device=torch.device("cpu")
                )
            gt_f = self.calc_feature(
                betas = motion["betas"][:1, ctx_seq_len:gt_seq_len],
                T_world_root = motion["T_world_root"][:1, ctx_seq_len:gt_seq_len],
                body_joint_rotations = motion["body_joint_rotations"][:1, ctx_seq_len:gt_seq_len],
                hand_joint_rotations = motion["hand_joint_rotations"][:1, ctx_seq_len:gt_seq_len],
                # device=torch.device("cpu"),
            )
            
            for k0 in keys0:
                for k1 in keys1: 
                    gt_feature[k0][k1].append(gt_f[k0][k1])
                    gen_feature[k0][k1].append(gen_f[k0][k1])
            for k0 in keys0_:
                gt_feature[k0].append(gt_f[k0])
                gen_feature[k0].append(gen_f[k0])

        person_num = motion_list[0]['betas'].shape[2]
            
        return gt_feature, gen_feature, person_num# , device, person_num

    @torch.inference_mode()
    def calc_feature(
        self,
        betas: Float[Tensor, "B T P 10"],
        T_world_root: Float[Tensor, "B T P 7"],
        body_joint_rotations: Float[Tensor, "B T P 21 4"],
        hand_joint_rotations: Float[Tensor, "B T P 30 4"] | None = None,
        device: torch.device | None = None,
    ) -> Dict[str, Dict[str, Float[Tensor, "B T P D"]]]:
        B, T, P, _ = betas.shape
        org_device = betas.device
        if device is not None and betas.device != device:
            print(f'org_device: {org_device}, device: {device}')
            betas = betas.to(device)
            T_world_root = T_world_root.to(device)
            body_joint_rotations = body_joint_rotations.to(device)
            if hand_joint_rotations is not None:
                hand_joint_rotations = hand_joint_rotations.to(device)
            self.smpl_model = self.smpl_model.to(device)
        else:
            self.smpl_model = self.smpl_model.to(org_device)
            device = org_device

        smpl_shaped = self.smpl_model.with_shape(betas)
        smpl_posed = smpl_shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_joint_rotations,
            left_hand_quats=None if hand_joint_rotations is None else hand_joint_rotations[..., :15, :],
            right_hand_quats=None if hand_joint_rotations is None else hand_joint_rotations[..., 15:, :],
        )

        T_world_joint = smpl_posed.Ts_world_joint
        
        # # remove the initial translation of the root joint
        # init_root_pos = T_world_root[:, 0, :, 4:].mean(dim=1)
        # init_root_pos[..., 1] = 0. # remove the vertical translation
        
        body_pos = T_world_joint[..., :21, 4:] #- init_root_pos[:, None, None, None, :]
        hand_pos = T_world_joint[..., -30:, 4:] #- init_root_pos[:, None, None, None, :]

        body_pose_wo_can = body_pos.clone()
        hand_pose_wo_can = hand_pos.clone()
        body_pose_wo_can[..., [0, 2]] = T_world_root[:, :, :, None, [4, 6]]
        hand_pose_wo_can[..., [0, 2]] = T_world_root[:, :, :, None, [4, 6]]

        body_pos_wo_root = T_world_joint[..., :21, 4:] - T_world_root[:, :, :, None, 4:]
        hand_pos_wo_root = T_world_joint[..., -30:, 4:] - T_world_root[:, :, :, None, 4:]

        # feat in world frame
        body_feat = body_pos.reshape(B, T, P, -1)
        hand_feat = hand_pos.reshape(B, T, P, -1)
        body_vel_feat = torch.diff(body_feat, dim=1)*self.fps
        hand_vel_feat = torch.diff(hand_feat, dim=1)*self.fps
        # body_vel_feat = (body_feat[:, 2:, :, :] - body_feat[:, :-2, :, :]) / 2 * self.fps
        # hand_vel_feat = (hand_feat[:, 2:, :, :] - hand_feat[:, :-2, :, :]) / 2 * self.fps


        # feat wo canonical (fid/div)
        body_feat_wo_can = body_pose_wo_can.reshape(B, T, P, -1)
        hand_feat_wo_can = hand_pose_wo_can.reshape(B, T, P, -1)
        body_vel_feat_wo_can = torch.diff(body_feat_wo_can, dim=1)*self.fps
        hand_vel_feat_wo_can = torch.diff(hand_feat_wo_can, dim=1)*self.fps
        # body_vel_feat_wo_can = (body_feat_wo_can[:, 2:, :, :] - body_feat_wo_can[:, :-2, :, :]) / 2 * self.fps
        # hand_vel_feat_wo_can = (hand_feat_wo_can[:, 2:, :, :] - hand_feat_wo_can[:, :-2, :, :]) / 2 * self.fps

        # feat wo root (mpjpe/mpjve)
        body_feat_wo_root = body_pos_wo_root.reshape(B, T, P, -1)
        hand_feat_wo_root = hand_pos_wo_root.reshape(B, T, P, -1)
        body_vel_feat_wo_root = torch.diff(body_feat_wo_root, dim=1)*self.fps
        hand_vel_feat_wo_root = torch.diff(hand_feat_wo_root, dim=1)*self.fps
        # body_vel_feat_wo_root = (body_feat_wo_root[:, 2:, :, :] - body_feat_wo_root[:, :-2, :, :]) / 2 * self.fps
        # hand_vel_feat_wo_root = (hand_feat_wo_root[:, 2:, :, :] - hand_feat_wo_root[:, :-2, :, :]) / 2 * self.fps

        return {
            "body": {
                "g": body_feat.to(org_device), "k": body_vel_feat.to(org_device), 
                "g_woc": body_feat_wo_can.to(org_device), "k_woc": body_vel_feat_wo_can.to(org_device), 
                "g_wor": body_feat_wo_root.to(org_device), "k_wor": body_vel_feat_wo_root.to(org_device),
            },
            "hand": {
                "g": hand_feat.to(org_device), "k": hand_vel_feat.to(org_device), 
                "g_woc": hand_feat_wo_can.to(org_device), "k_woc": hand_vel_feat_wo_can.to(org_device), 
                "g_wor": hand_feat_wo_root.to(org_device), "k_wor": hand_vel_feat_wo_root.to(org_device),
            },
            "root_pos": T_world_root[..., 4:].to(org_device),
        }

    @torch.inference_mode()
    def calc_mu_sigma(
        self,
        raw_feature: List[Float[Tensor, "S T P D"]],
        mode: Literal["solo", "multi"] = "solo",
        use_double: bool = True,
    ) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        B = len(raw_feature)
        S, T, P, D = raw_feature[0].shape
        feat_dim = P*D if mode == "multi" else D
        feature = torch.cat(raw_feature, dim=1)
        feature = feature.reshape(-1, feat_dim)

        if use_double:
            feature = feature.to(torch.float64)
        
        mu = feature.mean(dim=0)
        sigma = torch.cov(feature.T)
        return feature, mu, sigma

    @torch.inference_mode()
    def calc_mu_sigma_seq(
        self,
        raw_feature: List[Float[Tensor, "S T P D"]],
        seq: int,
        stride: int,
        mode: Literal["solo", "multi"] = "solo",
        use_double: bool = True,
    ) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        B = len(raw_feature)
        S, _, P, D = raw_feature[0].shape
        feat_dim = P*D*seq if mode == "multi" else D*seq
        raw_feature_list = []
        for i in range(len(raw_feature)):
            if raw_feature[i].shape[1] < seq:
                continue
            w = raw_feature[i].unfold(dimension=1, size=seq, step=stride)
            raw_feature_list.append(w)

        feature = torch.cat(raw_feature_list, dim=1)
        feature = feature.reshape(-1, feat_dim)

        if use_double:
            feature = feature.to(torch.float64)
        
        mu = feature.mean(dim=0)
        sigma = torch.cov(feature.T)
        return feature, mu, sigma

    @torch.inference_mode()
    def calc_mean_per_joint_error(
        self,
        gt_feature_list: List[Float[Tensor, "1 Tgt P D"]],
        gen_feature_list: List[Float[Tensor, "S Tgen P D"]],
        use_double: bool = True,
    ) -> float:

        error_list = []
        best_error_list = []
        for i in range(len(gt_feature_list)):
            gt_feat = gt_feature_list[i]
            gen_feat = gen_feature_list[i]
            T = min(gen_feat.shape[1], gt_feat.shape[1])
            gen_feat = gen_feat[:, :T]
            gt_feat = gt_feat[:, :T]
            S = gen_feat.shape[0]
            diff = gt_feat.expand(S, -1, -1, -1) - gen_feat
            if use_double:
                diff = diff.to(torch.float64)
            error = torch.linalg.vector_norm(diff.reshape(S, -1, 3), dim=-1)
            sample_error = error.mean(dim=1)
            best_error = torch.cummin(sample_error, dim=0).values
            error_list.append(error.reshape(-1))
            best_error_list.append(best_error)

        error_all = torch.cat(error_list, dim=0)
        best_error_all = torch.stack(best_error_list, dim=0)
        return error_all.mean().item(), best_error_all.mean(dim=0).cpu().tolist()


    @torch.inference_mode()
    def _symmetrize_psd(
        self,
        M: Float[Tensor, "D D"], 
        eps: float
    ) -> torch.Tensor:
        M = (M + M.T) * 0.5
        tau = eps * float(torch.trace(M).abs().item() + 1.0)
        I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        return M + tau * I

    @torch.inference_mode()
    def calc_frechet_distance(
        self,
        mu1: Float[Tensor, "D"],
        sigma1: Float[Tensor, "D D"],
        mu2: Float[Tensor, "D"],
        sigma2: Float[Tensor, "D D"],
        eps: float = 1e-6, 
        use_double: bool = True
    ) -> float:
        if use_double:
            mu1 = mu1.to(torch.float64)
            mu2 = mu2.to(torch.float64)
            sigma1 = sigma1.to(torch.float64)
            sigma2 = sigma2.to(torch.float64)

        S1 = self._symmetrize_psd(sigma1, eps)
        S2 = self._symmetrize_psd(sigma2, eps)

        diff2 = float(torch.dot(mu1 - mu2, mu1 - mu2))
        trS1 = float(torch.trace(S1))
        trS2 = float(torch.trace(S2))

        e1, U1 = torch.linalg.eigh((S1 + S1.T) * 0.5)
        e1 = torch.clamp(e1, min=0)
        S1h = (U1 * e1.clamp(min=0).sqrt().unsqueeze(-2)) @ U1.T
        # e1_sqrt = e1.clamp_min(0).sqrt()
        # S1h = U1 @ (e1_sqrt.unsqueeze(-1) * U1.T)

        mid = (S1h @ S2 @ S1h)
        mid = (mid + mid.T) * 0.5
        lam = torch.linalg.eigvalsh(mid)
        lam = torch.clamp(lam, min=0)
        tr_covmean = float(torch.sqrt(lam).sum())

        fid = diff2 + trS1 + trS2 - 2.0 * tr_covmean

        # if fid < 0 and fid > -1e-3:
        #     fid = 0.0
        return fid

    @torch.inference_mode()
    def calc_diversity(
        self,
        feature_mat: Float[Tensor, "N D"],
        diversity_times: int = 300, 
        generator: torch.Generator | None = None,
        use_double: bool = True,
    ) -> float:
        num_samples = feature_mat.shape[0]
        device = feature_mat.device
        
        if use_double:
            feature_mat = feature_mat.to(torch.float64)
        
        if num_samples < diversity_times:
            raise ValueError(f"activation.shape[0] (= {num_samples}) must be >= diversity_times (= {diversity_times})")

        idx1 = torch.randperm(num_samples, device=device, generator=generator)[:diversity_times]
        idx2 = torch.randperm(num_samples, device=device, generator=generator)[:diversity_times]

        dist_all = feature_mat[idx1] - feature_mat[idx2]
        dist_mean = dist_all.mean(dim=0)
        dist_var = torch.linalg.vector_norm(dist_all - dist_mean, dim=1).mean()
        return dist_var.item()

        # dist = torch.linalg.vector_norm(feature_mat[idx1] - feature_mat[idx2], dim=1)
        # return dist.mean().item()

    @torch.inference_mode()
    def calc_sample_diversity(
        self,
        feature_mat: Float[Tensor, "N D"],
        sample_num: int = 10,
        diversity_times: int = 300, 
        generator: torch.Generator | None = None,
        use_double: bool = True,
    ) -> float:
        num_samples = feature_mat.shape[0]
        device = feature_mat.device
        timestep_num = num_samples // sample_num

        rand_timestep = torch.randint(0, timestep_num, (diversity_times,), device=device, generator=generator)
        rand_sample1 = torch.randint(0, sample_num, (diversity_times,), device=device, generator=generator)
        rand_sample2 = torch.randint(0, sample_num, (diversity_times,), device=device, generator=generator)
        idx1 = rand_timestep * sample_num + rand_sample1
        idx2 = rand_timestep * sample_num + rand_sample2
        
        if use_double:
            feature_mat = feature_mat.to(torch.float64)
        
        if num_samples < diversity_times:
            raise ValueError(f"activation.shape[0] (= {num_samples}) must be >= diversity_times (= {diversity_times})")

        dist_all = feature_mat[idx1] - feature_mat[idx2]
        dist_mean = dist_all.mean(dim=0)
        dist_var = torch.linalg.vector_norm(dist_all - dist_mean, dim=1).mean()
        return dist_var.item()

        # dist = torch.linalg.vector_norm(feature_mat[idx1] - feature_mat[idx2], dim=1)
        # return dist.mean().item()


    @torch.inference_mode()
    def calc_sample_diversity_new(
        self,
        feature_list: List[Float[Tensor, "S T P D"]],
        is_seq: bool = False,
        use_double: bool = True,
    ):
        var_total = 0
        for feature in feature_list:
            S, T, P, D = feature.shape
            if use_double:
                feature = feature.to(torch.float64)
            if is_seq:
                T_ = feature.shape[1] // self.fps * self.fps
                if T_ <= 0:
                    continue
                var_total += torch.var(feature[:, :T_].reshape(S, -1, 63*self.fps), dim=0).mean(dim=0).sum()
            else:
                var_total += torch.var(feature.reshape(S, -1, 63), dim=0).mean(dim=0).sum()

        var_total /= len(feature_list)
        return var_total.item()


    @torch.inference_mode()
    def calc_person_correlation(
        self,
        feature_list: List[Float[Tensor, "S T P D"]],
        use_double: bool = True,
    ):
        B = len(feature_list)
        _, _, P, D = feature_list[0].shape
        feature = torch.cat(feature_list, dim=1)
        feature = feature.reshape(-1, P, D)
        corr_list = []
        
        if use_double:
            feature = feature.to(torch.float64)
        
        for p in range(P):
            for q in range(p+1, P):
                feature0 = feature[:, p]
                feature1 = feature[:, q]
                if torch.var(feature0) == 0 or torch.var(feature1) == 0:
                    corr_list.append(0.0)
                else:
                    try:
                        corr = torchmetrics.PearsonCorrCoef(num_outputs=D).to(feature.device)
                        corr.update(feature0, feature1)
                        corr_list.append(corr.compute().mean().item())
                    except:
                        corr_list.append(0.0)
                        print('fail corr')
        return torch.tensor(corr_list).mean().item()            

    @torch.inference_mode()
    def _segseg_distance_batch(
        self,
        p1: Float[Tensor, "... 3"],
        q1: Float[Tensor, "... 3"],
        p2: Float[Tensor, "... 3"],
        q2: Float[Tensor, "... 3"],
        eps: float = 1e-8,
    ) -> Float[Tensor, "..."]:
        u = q1 - p1
        v = q2 - p2
        w = p1 - p2

        a = (u * u).sum(dim=-1)
        b = (u * v).sum(dim=-1)
        c = (v * v).sum(dim=-1)
        d = (u * w).sum(dim=-1)
        e = (v * w).sum(dim=-1)

        D = a * c - b * b
        zero = torch.zeros_like(D)
        one  = torch.ones_like(D)

        sN = torch.where(D < eps, zero, (b * e - c * d))
        sD = torch.where(D < eps, one,  D)
        tN = torch.where(D < eps, e,    (a * e - b * d))
        tD = torch.where(D < eps, c,    D)

        mask = sN < 0
        sN = torch.where(mask, zero, sN)
        tN = torch.where(mask, e,    tN)
        tD = torch.where(mask, c,    tD)

        mask = sN > sD
        sN = torch.where(mask, sD,   sN)
        tN = torch.where(mask, e + b, tN)
        tD = torch.where(mask, c,    tD)

        mask = tN < 0
        tN = torch.where(mask, zero, tN)
        sN = torch.where(mask & (-d < 0), zero, sN)
        sN = torch.where(mask & (-d > a), sD,  sN)
        cond_mid = mask & ~(-d < 0) & ~(-d > a)
        sN = torch.where(cond_mid, -d, sN)
        sD = torch.where(cond_mid, a,  sD)

        mask = tN > tD
        tN = torch.where(mask, tD, tN)
        x = -d + b
        sN = torch.where(mask & (x < 0), zero, sN)
        sN = torch.where(mask & (x > a), sD,  sN)
        cond_mid = mask & ~(x < 0) & ~(x > a)
        sN = torch.where(cond_mid, x, sN)
        sD = torch.where(cond_mid, a, sD)

        sc = torch.where(sN.abs() < eps, zero, sN / sD)
        tc = torch.where(tN.abs() < eps, zero, tN / tD)

        dP = w + sc[..., None] * u - tc[..., None] * v
        return dP.norm(dim=-1)


    def _default_capsule_defs(self) -> List[Tuple[str, int, int, float]]:
        return [
            ("torso_lower",      0,  3, 0.12),
            ("torso_middle",     3,  6, 0.11),
            ("torso_upper",      6,  9, 0.10),
            ("neck",             9, 12, 0.06),
            ("head",            12, 15, 0.09),
            ("left_upper_arm",  16, 18, 0.05),
            ("left_lower_arm",  18, 20, 0.04),
            ("right_upper_arm", 17, 19, 0.05),
            ("right_lower_arm", 19, 21, 0.04),
            ("left_upper_leg",   1,  4, 0.07),
            ("left_lower_leg",   4,  7, 0.06),
            ("right_upper_leg",  2,  5, 0.07),
            ("right_lower_leg",  5,  8, 0.06),
        ]


    @torch.inference_mode()
    def _build_capsule_endpoints_tensor(
        self,
        joints_3d: Float[Tensor, "N P J 3"],
        root_pos: Float[Tensor, "N P 3"],
        capsule_defs: List[Tuple[str, int, int, float]],
        indices_are_one_based: bool = True,
    ) -> Tuple[Float[Tensor, "N P C 3"], Float[Tensor, "N P C 3"], Float[Tensor, "C"], List[str]]:
        device = joints_3d.device
        dtype  = joints_3d.dtype
        N, P, J, _ = joints_3d.shape

        names, j1, j2, radii = zip(*capsule_defs)
        C = len(names)

        i1 = torch.tensor(j1, device=device, dtype=torch.long)
        i2 = torch.tensor(j2, device=device, dtype=torch.long)
        r  = torch.tensor(radii, device=device, dtype=dtype)

        # 0=root
        k1 = torch.where(i1 == 0, torch.zeros_like(i1),
                        (i1 - 1) if indices_are_one_based else i1)
        k2 = torch.where(i2 == 0, torch.zeros_like(i2),
                        (i2 - 1) if indices_are_one_based else i2)

        k1c = k1.clamp(0, max(J - 1, 0))
        k2c = k2.clamp(0, max(J - 1, 0))

        joints_i1 = joints_3d.index_select(dim=2, index=k1c)  # (N,P,C,3)
        joints_i2 = joints_3d.index_select(dim=2, index=k2c)

        root_exp = root_pos.unsqueeze(2).expand(N, P, C, 3)
        use_root_1 = (i1 == 0).view(1, 1, C, 1)
        use_root_2 = (i2 == 0).view(1, 1, C, 1)

        p = torch.where(use_root_1, root_exp, joints_i1)
        q = torch.where(use_root_2, root_exp, joints_i2)

        # invalid indices
        invalid1 = (i1 != 0) & ((k1 < 0) | (k1 >= J))
        invalid2 = (i2 != 0) & ((k2 < 0) | (k2 >= J))
        invalid = (invalid1 | invalid2).to(device)

        if invalid.any():
            r = r * (~invalid).to(dtype)

        return p, q, r, names


    @torch.inference_mode()
    def calc_penetration_metric_fast(
        self,
        feature_list: List[Float[Tensor, "S T P ..."]],
        root_pos_list: List[Float[Tensor, "S T P 3"]],
        ignore_pairs: List[Tuple[str, str]]| None=None,
        ignore_distance: float = 0.05,
        indices_are_one_based: bool = True,
        capsule_defs: List[Tuple[str, int, int, float]]| None=None,
        chunk_size: int | None = None,
    ) -> Dict[str, float]:
        if capsule_defs is None:
            capsule_defs = self._default_capsule_defs()
        if ignore_pairs is None:
            ignore_pairs = [
                ("left_lower_arm", "right_lower_arm"),
                ("right_lower_arm", "left_lower_arm"),
                ("left_lower_arm", "left_lower_arm"),
                ("right_lower_arm", "right_lower_arm")
            ]

        dev= feature_list[0].device
        dt = feature_list[0].dtype
        S, _, P, *_ = feature_list[0].shape
        feature = torch.cat(feature_list, dim=1)
        root_pos = torch.cat(root_pos_list, dim=1)
        J = 21

        joints = feature.reshape(-1, P, J, 3).to(device=dev, dtype=dt)
        roots  = root_pos.reshape(-1, P, 3).to(device=dev, dtype=dt)
        N = joints.shape[0]
        if N > 1000000:
            joints = joints[::10]
            roots = roots[::10]
            N = joints.shape[0]

        # pair (M,2)
        ppl_pairs = torch.combinations(torch.arange(P, device=dev), r=2)  # (M,2)
        M = ppl_pairs.shape[0]
        A_idx = ppl_pairs[:, 0]
        B_idx = ppl_pairs[:, 1]

        # ignore mask (C,C)
        p_all, q_all, r_all, names = self._build_capsule_endpoints_tensor(
            joints[:1], roots[:1], capsule_defs, indices_are_one_based
        )
        C = r_all.numel()
        ignore_mask = torch.zeros((C, C), dtype=torch.bool, device=dev)
        name2idx = {n: i for i, n in enumerate(names)}
        for a, b in ignore_pairs:
            if a in name2idx and b in name2idx:
                ia, ib = name2idx[a], name2idx[b]
                ignore_mask[ia, ib] = True

        total_frames = N
        frames_with_any = 0

        pos_pair_count = 0
        sum_total_pen  = torch.zeros((), device=dev, dtype=dt)

        pos_max_count  = 0
        sum_max_depth  = torch.zeros((), device=dev, dtype=dt)

        overall_max_depth = torch.zeros((), device=dev, dtype=dt)

        if chunk_size is None or chunk_size <= 0:
            chunk_size = N if N > 0 else 1

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            joints_ch = joints[start:end]  # (n,P,J,3)
            roots_ch  = roots[start:end]   # (n,P,3)
            n = joints_ch.shape[0]
            if n == 0:
                continue

            p_end, q_end, r, _ = self._build_capsule_endpoints_tensor(
                joints_ch, roots_ch, capsule_defs, indices_are_one_based
            )  # (n,P,C,3), (C,)

            pA = p_end[:, A_idx]  # (n,M,C,3)
            qA = q_end[:, A_idx]
            pB = p_end[:, B_idx]
            qB = q_end[:, B_idx]

            P1 = pA.unsqueeze(3)
            Q1 = qA.unsqueeze(3)
            P2 = pB.unsqueeze(2)
            Q2 = qB.unsqueeze(2)

            # minimum distance (n,M,C,C)
            d = self._segseg_distance_batch(P1, Q1, P2, Q2)

            # depth = (ri+rj) - d
            rsum = r.view(C, 1) + r.view(1, C)         # (C,C)
            depth = (rsum - d).clamp_min(0)            # (n,M,C,C)

            # ignore mask
            if ignore_mask.any():
                depth = depth.masked_fill(ignore_mask.view(1, 1, C, C), 0)

            # ignore distance
            if ignore_distance > 0:
                depth = torch.where(depth >= ignore_distance, depth, torch.zeros_like(depth))

            # pair total/maximum (n,M)
            total_per_pair = depth.sum(dim=(2, 3))
            max_per_pair   = depth.amax(dim=(2, 3))

            positive = total_per_pair > 0                      # (n,M)
            frames_with_any += int(positive.any(dim=1).sum().item())

            # avg_total_penetration
            if positive.any():
                sum_total_pen += total_per_pair[positive].sum()
                pos_pair_count += int(positive.sum().item())
                sum_max_depth += max_per_pair[positive].sum()
                pos_max_count += int(positive.sum().item())

            # overall_max_depth
            if max_per_pair.numel() > 0:
                overall_max_depth = torch.maximum(
                    overall_max_depth, max_per_pair.max()
                )

        avg_total_pen = (sum_total_pen / pos_pair_count) if pos_pair_count > 0 else torch.zeros((), device=dev, dtype=dt)
        avg_max_depth = (sum_max_depth / pos_max_count) if pos_max_count > 0 else torch.zeros((), device=dev, dtype=dt)
        penetration_ratio = (frames_with_any / total_frames) if total_frames > 0 else 0.0

        return {
            'avg_total_penetration': float(avg_total_pen),
            'avg_max_depth': float(avg_max_depth),
            'overall_max_depth': float(overall_max_depth),
            'penetrating_frames': int(frames_with_any),
            'total_frames': int(total_frames),
            'penetration_ratio': float(penetration_ratio),
        }

    @torch.inference_mode()
    def calc_foot_skate_metric(
        self,
        feature_list: List[Float[Tensor, "S T P ..."]]
    ):
        B = len(feature_list)
        # S, T, P, _ = feature_list[0].shape
        foot_height_thresh = -0.01
        foot_vel_thresh = 0.05
        foot_sliding_vel = []
        foot_sliding_frames = 0
        total_frames = 0
        for feature in feature_list:
            S, T, P, _ = feature.shape
            total_frames += S*T*P
            joint_position = feature.reshape(S, T, P, 21, 3)
            foot_position = joint_position[..., [9, 10], :]
            foot_vel = torch.diff(foot_position[..., [0, 2]], dim=1) * self.fps
            foot_contact = foot_position[:, :, :, :, 1] < foot_height_thresh
            foot_contact = torch.logical_and(foot_contact[:, 1:], foot_contact[:, :-1])
            foot_vel = torch.linalg.norm(foot_vel, dim=-1)
            foot_sliding_idx = torch.logical_and(foot_contact, foot_vel > foot_vel_thresh)
            foot_sliding_vel.append(foot_vel[foot_sliding_idx])
            foot_sliding_frames += (foot_sliding_idx.sum(dim=-1) > 0).sum()
        
        foot_sliding_ratio = foot_sliding_frames / total_frames
        avg_foot_sliding_vel = torch.mean(torch.cat(foot_sliding_vel))
        
        return {
            'foot_sliding_ratio': foot_sliding_ratio.item(),
            'avg_foot_sliding_vel': avg_foot_sliding_vel.item(),
        }
        
        # joint_position = feature.reshape(B*S, T, P, 21, 3)
        # foot_position = joint_position[..., [9, 10], :]
        # foot_vel = torch.diff(foot_position[..., [0, 2]], dim=1) * self.fps
        # foot_contact = foot_position[:, :, :, :, 1] < foot_height_thresh
        # foot_contact = torch.logical_and(foot_contact[:, 1:], foot_contact[:, :-1])
        # foot_sliding_vel = torch.linalg.norm(foot_vel, dim=-1)
        # foot_sliding_idx = torch.logical_and(foot_contact, foot_sliding_vel > foot_vel_thresh)
        # avg_foot_sliding_vel = torch.mean(foot_sliding_vel[foot_sliding_idx])
        # foot_sliding_frames = foot_sliding_idx.sum(dim=-1)
        # foot_sliding_ratio = torch.sum(foot_sliding_frames > 0) / (B*S*T*P)
        return {
            'foot_sliding_ratio': foot_sliding_ratio.item(),
            'avg_foot_sliding_vel': avg_foot_sliding_vel.item(),
        }


   


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--whole_seq", action="store_true", default=False)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--test_set", type=str, default=None)
    args = parser.parse_args()
    em = EvalMotion()
    em.eval_from_dir(
        Path(args.data_dir), 
        fps=args.fps, 
        sample_num=args.sample_num, 
        is_whole_seq=args.whole_seq, 
        seq_len=args.seq_len, 
        test_set_path=Path(args.test_set) if args.test_set is not None else None, 
        device=torch.device("cuda" if args.cuda else "cpu"))
