from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import monai
from Train import NpzDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Inference import compute_dice, finetune_model_predict
from skimage import io


# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iou(pred, target):

    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    pred = torch.sigmoid(pred)
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    pred = pred.to(torch.int32)
    target = target.to(torch.int32)
    inter = (pred & target).sum(dim=(1, 2))
    union = (pred | target).sum(dim=(1, 2)) + 1e-5
    iou = 1 - (inter / union)

    return iou.mean()


def predict(sam_model, i, train):

    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    if train:
        npz = np.load("data/Train/custom.npz")
    else:
        npz = np.load(f"data/custom_vit_b/custom.npz")
    ori_imgs = npz['imgs']
    ori_gts = npz['gts']
    ori_number = npz['number']
    boundary = npz['boundary']

    sam_segs = []
    sam_bboxes = []
    sam_dice_scores = []
    medsam_seg_probs = []
    pred = []
    for img_id, ori_img in tqdm(enumerate(ori_imgs)):
        # get bounding box from mask
        gt2D = ori_gts[img_id]
        bboundary = boundary[img_id]

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = np.array([x_min, y_min, x_max, y_max])
        seg_mask, medsam_seg_prob = finetune_model_predict(ori_img, bbox, bboundary, sam_trans, sam_model, device=device)
        sam_segs.append(seg_mask)
        sam_bboxes.append(bbox)
        medsam_seg_probs.append(medsam_seg_prob)
        sam_dice_scores.append(compute_dice(seg_mask > 0, gt2D > 0))
        miou = iou(torch.tensor(seg_mask), torch.tensor(gt2D))
        if miou.item() > 0.90:
            pred.append(0)
        else:
            pred.append(1)
    os.makedirs(f"output-85/model_{i}", exist_ok=True)
    np.savez_compressed(os.path.join(f"output-85/model_{i}/custom.npz"), medsam_segs=sam_segs, gts=ori_gts, number=ori_number, sam_bboxes=sam_bboxes, medsam_seg_prob=medsam_seg_probs)
    return pred, medsam_seg_probs, sam_bboxes, ori_number

def train(train_dataloader, i):
    
    work_dir = './work_dir_cod'
    task_name = 'COMPromter_iter5'
    # prepare SAM model
    model_type = 'vit_b'
    checkpoint = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'
    model_save_path = os.path.join(work_dir, task_name, i)
    os.makedirs(model_save_path, exist_ok=True)
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

    sam_model.train()
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = 5
    losses = []
    best_loss = 1e10
    
    for epoch in range(num_epochs+1):
        epoch_loss = 0

        for step, (image_embedding, gt2D, boxes, boundary) in enumerate(tqdm(train_dataloader)):

            with torch.no_grad():
                box_np = boxes.numpy()
                sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
                box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                boundary = torch.as_tensor(boundary, dtype=torch.float, device=device)
                image_embedding = torch.as_tensor(image_embedding, dtype=torch.float, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)
                sparse_embeddings_box, dense_embeddings_box = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None
                )

                sparse_embeddings_boundary, dense_embeddings_boundary = sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=boundary
                )


            high_frequency = sam_model.DWT(image_embedding)
            dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_boundary, dense_embeddings_box,
                                                            high_frequency, sparse_embeddings_box)

            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            loss = seg_loss(mask_predictions, gt2D.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        losses.append(epoch_loss)
        print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
        # save the latest model checkpoint
        if epoch >= 250 and epoch % 10 == 0:
            torch.save(sam_model.state_dict(), os.path.join(model_save_path, str(epoch) + 'sam_model_{i}.pth'))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(model_save_path, f'sam_model_best_{i}.pth')
            torch.save(sam_model.state_dict(), save_path)

    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(model_save_path, 'train_loss.png'))
    plt.close()
    return sam_model, save_path

def main():
    n_learners = 4

    alphas = []
    models = []
    npz_tr_path = 'data/Train'
    train_dataset = NpzDataset(npz_tr_path)
    for i in range(n_learners):

        if i == 0:
            len_data = train_dataset.__len__()
            weights = torch.ones(len_data) / len_data  # init weight
            label = torch.ones(len_data)

        samples_weight = torch.from_numpy(np.array(weights))
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)

        model, save_path = train(train_dataloader, str(i))  
        
        # predicts, sam_segs, _, ori_number = predict(model, str(i), True) 
        predicts, medsam_seg_probs, _, ori_number = predict(model, str(i), True) 
        predicts = torch.tensor(predicts)
        error = (predicts != label).float().mean().item()
        print(error)
        error = max(min(error, 1 - 1e-10), 1e-10) 
        alpha = 0.5 * np.log((1 - error) / error)
        weights *= torch.exp(alpha * (predicts != label).float())
        weights /= weights.sum() 

        alphas.append(alpha)
        models.append(model)
        print(alphas)

    sum_alphas = sum(alphas)
    normalized_alphas = [alpha / sum_alphas for alpha in alphas]
    normalized_alphas = torch.tensor(normalized_alphas)
    print(normalized_alphas)
    for ind, model_fin in enumerate(models):
        
        predict_, medsam_seg_probs, _, ori_number = predict(model_fin, str(ind), False) 
        medsam_seg_probs = torch.tensor(medsam_seg_probs)
        if ind == 0:
            probs = torch.zeros_like(medsam_seg_probs)
        probs += normalized_alphas[ind] * medsam_seg_probs

    for i_, img in enumerate(probs):
        if torch.is_tensor(img):
            img = img.numpy()
        img = np.clip(img, 0, 1)
        img_uint8 = (img * 255).astype(np.uint8)
        img_path = "fin_output-5iter/" + ori_number[i_]
        img_path = img_path.replace("jpg", "png")
        if not os.path.exists("fin_output"):
            os.makedirs("fin_output-5iter", exist_ok=True)
        io.imsave(img_path, img_uint8)

    print("Images saved successfully.")

if __name__ == "__main__":
    main()