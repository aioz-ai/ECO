import os
import numpy as np
import argparse
import torch
import vis_utils
import trimesh
import open3d as o3d
from tqdm import tqdm
import eulerangles
import general_utils
import time
import sys
from ContactECO.eco import ECO, convert_me_to_torch, expand_me_tensor, cat_me_tensor
import MinkowskiEngine as ME 
import torch.nn.functional as F
from cka import gram_linear, cka
from general_utils import compute_recon_loss_posa, compute_recon_loss


def lcka_score(i, l):  # calculate CKA score of mask[i]
    score = 0 
    for j in range(len(l)): 
        if j != i: 
            score+= (1- cka(gram_linear(l[i]), gram_linear(l[j])))
    return score 


def cka_score(z, ver, ver_mask, model): # store cka score of each mask in a list 
    no_mask = model.num_mask 
    score = [] # list of cka score 
    for i in range(no_mask):
        # forward phase in posa decoder (till the first layer)  
        ver_i = torch.clone(ver) 
        ver_i[:,ver_mask[i],:] = 0
        ver_i = ver_i.permute(0,2,1) 
        ver_i = ME.to_sparse_all(ver_i)
        w_i = ver_i.F.shape[0]
        z_i = ME.to_sparse_all(z) 
        z_i = expand_me_tensor(z_i,1,w_i) 
        x_i = cat_me_tensor(ver_i,z_i)
        o_i = model.decoder.graphin(x_i) # first layer of model
        o_i = convert_me_to_torch(o_i)
        # reshape output to 2-dimension for CKA calculation
        h,w,c = o_i.shape 
        o_i = o_i.reshape(h*w,c)
        o_i = o_i.detach().cpu().numpy()
        score.append(o_i) 
    cka_score = [0]*no_mask
    for i in range(len(score)): 
        cka_score[i] = lcka_score(i,score) 
    return torch.tensor(cka_score).unsqueeze(-1).cuda()/(no_mask-1)

# combination of weight contr and CKA
def alpha_score(cka_score,  ver_contr, epsilon = 0.5): 
    return cka_score * epsilon + (1-epsilon)*ver_contr

# select k masks with highest score
def retrieve_top_k(alpha, k):
    val, ind = torch.topk(alpha, k, dim = 0)
    if k==1: return val, [ind.item()]
    return val,[x.item() for x in list(ind.squeeze())]



def list_mean(list):
    acc = 0.
    for item in list:
        acc += item
    return acc / len(list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to dataset dir")
    parser.add_argument("--load_model_path", type=str, default="",
                        help="checkpoint path to load")
    parser.add_argument("--seg_len", type=int, default=32,
                        help="the number of frames of each input sequence to POSA_temp")
    parser.add_argument("--scene_dir", type=str, default="",
                        help="the path to the scene mesh")
    parser.add_argument("--tpose_mesh_dir", type=str, default="../data/mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")
    parser.add_argument("--save_video", dest='save_video', action='store_const', const=True, default=False)
    parser.add_argument("--save_video_dir", type=str, default="../results/",
                        help="the path to save validation results")
    parser.add_argument("--cam_setting_path", type=str, default="./support_files/ScreenCamera_0.json",
                        help="the path to camera settings in open3d")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--single_seq_name", type=str, default="BasementSittingBooth_00142_01")
    parser.add_argument("--model_name", type=str, default="default_model",
                        help="The name of current temporal model")
    parser.add_argument("--use_pretrained", dest="use_pretrained", action="store_const", const=True, default=False)
    parser.add_argument("--do_train", dest='do_train', action='store_const', const=True, default=False)
    parser.add_argument("--do_valid", dest='do_valid', action='store_const', const=True, default=False)
    parser.add_argument("--visualize", dest="visualize", action='store_const', const=True, default=False)
    parser.add_argument("--jump_step", type=int, default=1)
    parser.add_argument("--save_result", dest='save_result', action='store_const', const=True, default=False)
    parser.add_argument("--save_result_dir", type=str, default="../predict_results/ECO")
    parser.add_argument("--single_frame", type=int, default=-1)

    parser.add_argument("--cka", dest='cka', action='store_const', const=True, default=False)
    parser.add_argument("--no_sparse_mask", type=int, default=3)
    parser.add_argument("--sparsity_ratio", type=float, default=0.9)
    parser.add_argument("--keep_mask", type=int, default=3)

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    scene_dir = args_dict['scene_dir']
    tpose_mesh_dir = args_dict['tpose_mesh_dir']
    ckpt_path = args_dict['load_model_path']
    seg_len = args_dict['seg_len']
    bs = args_dict['batch_size']
    save_video = args_dict['save_video']
    save_video_dir = args_dict['save_video_dir']
    if not os.path.isdir(save_video_dir):
        os.mkdir(save_video_dir)
    cam_path = args_dict['cam_setting_path']
    cam_idx = 0
    for char in cam_path:
        if char.isdigit():
            cam_idx = char
    print("Using camera ", cam_idx)
    single_seq_name = args_dict['single_seq_name']
    model_name = args_dict['model_name']
    use_pretrained = args_dict['use_pretrained']
    do_train = args_dict['do_train']
    do_valid = args_dict['do_valid']
    visualize = args_dict['visualize']
    jump_step = args_dict['jump_step']
    save_result = args_dict['save_result']
    save_result_dir = args_dict['save_result_dir']


    num_mask = args_dict['no_sparse_mask']
    sparsity = args_dict['sparsity_ratio']
    keep_mask = args_dict['keep_mask']
    cka_sim = args_dict['cka']
    seq_name_list = []
    iou_s_list = []
    f1_s_list = []
    recon_semantics_acc_list = []
    inconsistency_list = []
    num_obj_classes = 8
    seq_class_acc = [[] for _ in range(num_obj_classes)]
    single_frame = args_dict['single_frame']

    contacts_s_dir = os.path.join(data_dir, "semantics")
    vertices_can_dir = os.path.join(data_dir, "vertices_can")
    vertices_dir = os.path.join(data_dir, "vertices")
    if do_valid or do_train:
        vertices_file_list = os.listdir(vertices_dir)
        seq_name_list = [file_name.split('_verts')[0] for file_name in vertices_file_list]
    else:
        seq_name_list = [single_seq_name]
    device = torch.device("cuda")
    model = ECO(num_mask = num_mask, sparsity_ratio = sparsity, ds_us_dir=tpose_mesh_dir, use_semantics=True).to(device)
    checkpoint = torch.load(ckpt_path)
    print(checkpoint['total_semantics_recon_acc'])
    # load the mask
    model.vertices_mask = checkpoint['vertices_mask']
    model.load_state_dict(checkpoint['model_state_dict'])

    # weight contr after training
    ver_contr = checkpoint['weight_contr']#model.vertices_contr.weight.data 
    ver_contr = F.softmax(ver_contr, dim=0) 
    os.makedirs(save_result_dir, exist_ok=True)
    seq_name_list = ['N0Sofa_00034_02','N3OpenArea_03301_01', 'MPH112_00151_01', 'MPH112_00169_01', 'MPH11_00150_01', 'N3Office_00034_01']
    model.eval()
    with torch.no_grad():
        for seq_name in seq_name_list:
            plot_time = []
            save_seq_dir = os.path.join(save_video_dir, seq_name)
            os.makedirs(save_seq_dir, exist_ok=True)
            save_seq_model_dir = os.path.join(save_seq_dir, model_name)
            os.makedirs(save_seq_model_dir, exist_ok=True)
            output_image_dir = os.path.join(save_seq_model_dir, cam_path.split("/")[-1])
            os.makedirs(output_image_dir, exist_ok=True)
            print("Test scene: {}".format(seq_name))


            verts = torch.tensor(np.load(os.path.join(vertices_dir, seq_name + "_verts.npy"))).to(device)
            verts_can = torch.tensor(np.load(os.path.join(vertices_can_dir, seq_name + "_verts_can.npy"))).to(device)
            contacts_s = torch.tensor(np.load(os.path.join(contacts_s_dir, seq_name + "_cfs.npy"))).to(device)
            
            # Set up visualizer
            if save_video:
                visualizer = o3d.visualization.Visualizer()
                visualizer.create_window()

            if visualize or save_video:
                scene_name = seq_name.split("_")[0]
                test_scene_mesh_path = "{}/{}.ply".format(scene_dir, scene_name)
                scene = o3d.io.read_triangle_mesh(test_scene_mesh_path)
                down_sample_level = 2
                # tpose for getting face_arr
                tpose_mesh_path = os.path.join(tpose_mesh_dir, "mesh_{}.obj".format(down_sample_level))
                faces_arr = trimesh.load(tpose_mesh_path, process=False).faces

            # Loop over video frames to get predictions
            # Metrics for semantic labels
            
            max_frame = 256
            iou_s = 0
            f1_s = 0
            recon_semantics_acc = 0
            #inconsistency = 0
            class_acc_list = [[] for _ in range(num_obj_classes)]
            class_acc = dict()
            pred_list = []
            count = 0
            for i in tqdm(range(0, verts.shape[0], jump_step)):
                count += 1
                # if count > max_frame:
                #     count -= 1
                #     break
            
                if single_frame != -1:
                    if i != single_frame:
                        continue

                verts_batch = verts[i].unsqueeze(0)
                verts_can_batch = verts_can[i]
                # if (use_pretrained):
                #     R_can = torch.tensor(eulerangles.euler2mat(np.pi, np.pi, -np.pi/2, 'syzx'), dtype=verts_can_batch.dtype, device=device)
                #     verts_can_batch = torch.matmul(R_can, verts_can_batch.t()).t()
                verts_can_batch = verts_can_batch.unsqueeze(0)

                # model prediction
                z = torch.tensor(np.random.normal(0, 1, (bs, 256)).astype(np.float32)).to(device)
                z = z.unsqueeze(-1)
                # cka_s = cka_score(z, verts_can_batch, model.vertices_mask,model) 
                # #alpha score for each mask
                # final_alpha = alpha_score(cka_s, ver_contr, epsilon = 0.5)
                # # select the masks with highest score: return score and mask idx
                # val, indices_retrieval = retrieve_top_k(final_alpha,k=keep_mask)
                # # tune the score
                # val = F.softmax(val, dim = 0) 

                z = ME.to_sparse_all(z)
                
                pr_cf_list = []
                # forward through selected masks
                for i in range(num_mask): 
                    sparse_ver_i = torch.clone(verts_can_batch)
                    sparse_ver_i[:,model.vertices_mask[i],:] = 0
                    # sparse_ver_i[:,model.vertices_mask[indices_retrieval[i]],:] = 0
                    ver_i = ver_contr[i][0]*(sparse_ver_i)
                    ver_i = ver_i.permute(0,2,1) 
                    ver_i = ME.to_sparse_all(ver_i)


                    #forward
                    prcf = model.decoder(z, ver_i)
                    prcf = convert_me_to_torch(prcf)
                    pr_cf_list.append(prcf)
                
                # sum all the results
                pred = torch.stack(pr_cf_list, dim=0).sum(dim=0)
                pred_frame = pred.squeeze()
                pred_list.append(pred_frame)
                contacts_s_onehot = torch.zeros(*contacts_s[i].cpu().shape, num_obj_classes, dtype=torch.float32).scatter_(-1,
                                                                                    contacts_s[i].cpu().unsqueeze(-1).type(torch.long),
                                                                                    1.).to(device)
                _, recon_semantics_acc_frame = compute_recon_loss_posa(contacts_s_onehot.unsqueeze(0), pred_frame.unsqueeze(0), **args_dict)
                # Calculate metrics for semantic labels
                contacts_s_frame = contacts_s[i].squeeze()
                pred_semantics = torch.argmax((pred_frame), dim=1)
                recon_semantics_acc += recon_semantics_acc_frame# torch.mean((pred_semantics == contacts_s_frame).to(torch.float32))
                for c in range(num_obj_classes):
                    c_idx = (contacts_s_frame == c)
                    if c_idx.sum() == 0:
                        continue
                    class_acc_list[c].append((pred_semantics[c_idx] == c).sum() / c_idx.sum())

                contacts_s_frame = general_utils.onehot_encode(contacts_s_frame, num_obj_classes, device)
                pred_semantics = general_utils.onehot_encode(pred_semantics, num_obj_classes, device)
                num_obj_classes_appeared = 0
                iou_s_cur = 0
                f1_s_cur = 0
                for class_idx in range(num_obj_classes):
                    if torch.sum(contacts_s_frame[:, class_idx] > 0):
                        num_obj_classes_appeared += 1
                        iou_s_cur += general_utils.compute_iou(contacts_s_frame[:, class_idx], pred_semantics[:, class_idx])
                        f1_s_cur += general_utils.compute_f1_score(contacts_s_frame[:, class_idx], pred_semantics[:, class_idx])
                iou_s += iou_s_cur / num_obj_classes_appeared
                f1_s += f1_s_cur / num_obj_classes_appeared
                
                
                # Visualization
                if visualize or save_video:
                    vis = []
                    vis += [scene]
                    vis += vis_utils.show_sample(verts_batch, pred, faces_arr, True)
                if visualize:
                    o3d.visualization.draw_geometries(vis)
                if save_video:
                    for geometry in vis:
                        visualizer.add_geometry(geometry)
                        visualizer.update_geometry(geometry)

                    ctr = visualizer.get_view_control()
                    parameters = o3d.io.read_pinhole_camera_parameters(cam_path)
                    ctr.convert_from_pinhole_camera_parameters(parameters)
                    visualizer.poll_events()
                    visualizer.update_renderer()
                    visualizer.capture_screen_image(os.path.join(output_image_dir, "frame_{:04d}.png".format(i)))
                    for geometry in vis:
                        visualizer.remove_geometry(geometry)
            if save_video:
                visualizer.destroy_window()

    
            pred_list = torch.stack(pred_list, dim=0)
            if save_result:
                save_result_path = os.path.join(save_result_dir, seq_name + ".npy")
                pred_npy = torch.argmax(pred_list, dim=-1).unsqueeze(-1).detach().cpu().numpy()
                np.save(save_result_path, pred_npy)
            vertsa = verts[::jump_step]
            vertsa = vertsa[:count]
            # inconsistency = general_utils.compute_consistency_metric(pred_list, vertsa)
            # inconsistency_list.append(inconsistency)

            for c in range(num_obj_classes):
                if len(class_acc_list[c]) == 0:
                    class_acc[c] = 1
                else:
                    class_acc[c] = list_mean(class_acc_list[c])
                seq_class_acc[c].append(class_acc[c])

            iou_s /= count; iou_s_list.append(iou_s)
            f1_s /= count; f1_s_list.append(f1_s)
            recon_semantics_acc /= count; recon_semantics_acc_list.append(recon_semantics_acc)
            f = open(os.path.join(save_seq_model_dir, "results.txt"), "w")
            f.write("IOU for semantic labels: {:.4f}".format(iou_s) + '\n')
            f.write("F1-score for semantic labels: {:.4f}".format(f1_s) + '\n')
            f.write("Reconstructed Semantics Accuracy: {:.4f}".format(recon_semantics_acc) + '\n')
            #f.write("Inconsistency score: {:.4f}".format(inconsistency) + '\n')
            f.write('\n')
            for c in range(num_obj_classes):
                f.write(f"Class {c} has acc: {class_acc[c]:.4f} \n")
            f.close()


    if do_valid:
        f = open(os.path.join(save_video_dir, "validation_results_{}.txt".format(model_name)), "w")
        f.write("Model use {} masks with sparsity {}, and keep {} mask(s) while testing. No CKA is considered\n".format(num_mask, sparsity, keep_mask))
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"IOU for {seq_name}: {iou_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"F1-score for {seq_name}: {f1_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"Reconstructed Semantics Accuracy for {seq_name}: {recon_semantics_acc_list[i]:.4f}\n")
        f.write("\n")
        # for i, seq_name in enumerate(seq_name_list):
        #     f.write(f"Inconsistency score for {seq_name}: {inconsistency_list[i]:.4f}\n")
        f.write("\n")
        f.write("IOU for semantic labels: {:.4f}".format(list_mean(iou_s_list)) + '\n')
        f.write("F1-score for semantic labels: {:.4f}".format(list_mean(f1_s_list)) + '\n')
        f.write("Reconstructed Semantics Accuracy: {:.4f}".format(list_mean(recon_semantics_acc_list)) + '\n')
        # f.write("Inconsistency Score: {:.4f}".format(list_mean(inconsistency_list)) + '\n')
        f.write("\n")
        for c in range(num_obj_classes):
            f.write(f"Acc for class {c}: {list_mean(seq_class_acc[c]):.4f} \n")
        f.close()

    if do_train:
        f = open(os.path.join(save_video_dir, "train_results_{}.txt".format(model_name)), "w")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"IOU for {seq_name}: {iou_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"F1-score for {seq_name}: {f1_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"Reconstructed Semantics Accuracy for {seq_name}: {recon_semantics_acc_list[i]:.4f}\n")
        f.write("\n")
        f.write("IOU for semantic labels: {:.4f}".format(list_mean(iou_s_list)) + '\n')
        f.write("F1-score for semantic labels: {:.4f}".format(list_mean(f1_s_list)) + '\n')
        f.write("Reconstructed Semantics Accuracy: {:.4f}".format(list_mean(recon_semantics_acc_list)) + '\n')
        f.write("\n")
        for c in range(num_obj_classes):
            f.write(f"Acc for class {c}: {list_mean(seq_class_acc[c]):.4f} \n")
        f.close()
