cuda=${1:-0}
init_model=init_nometal
init_iter=40000
model=radiogs_nometal
iter=20000

# air_baloons
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/air_baloons \
    -m outputs/Synthetic4Relight/air_baloons/$init_model --eval -w \
    --lambda_mask_entropy 0.05 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --lambda_normal_smooth 0.02 \
    # --train_sh_vol \
    # --radiosity_from_iter 2000 \
    # --lambda_radiosity 0.1

# chair
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/chair \
    -m outputs/Synthetic4Relight/chair/$init_model --eval -w \
    --lambda_mask_entropy 0.05 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_radiosity 0.1 \
    --lambda_normal_smooth 0.02

# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/hotdog \
    -m outputs/Synthetic4Relight/hotdog/$init_model --eval -w \
    --lambda_mask_entropy 0.05 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --lambda_normal_smooth 0.02 \
    # --train_sh_vol \
    # --radiosity_from_iter 2000 \
    # --lambda_radiosity 0.1 \
    
    

# jugs
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/Synthetic4Relight/jugs \
    -m outputs/Synthetic4Relight/jugs/$init_model --eval -w \
    --lambda_mask_entropy 0.05 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1
    
    
# air_baloons
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/Synthetic4Relight/air_baloons --eval \
    -m outputs/Synthetic4Relight/air_baloons/$model --iterations $iter \
    --start_checkpoint_refgs outputs/Synthetic4Relight/air_baloons/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 1.0 --lambda_roughness_smooth 0.5 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.0 --lambda_nvs 1.0  --back_culling --light_t_min 0.1 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/Synthetic4Relight/air_baloons/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_syn4.py -m outputs/Synthetic4Relight/air_baloons/$model  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_material_syn4.py -m outputs/Synthetic4Relight/air_baloons/$model \
    --albedo_rescale 1  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_syn4.py -m outputs/Synthetic4Relight/air_baloons/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 1 -e light   --back_culling


# chair
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/Synthetic4Relight/chair --eval \
    -m outputs/Synthetic4Relight/chair/$model --iterations $iter \
    --start_checkpoint_refgs outputs/Synthetic4Relight/chair/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 1.0 --lambda_roughness_smooth 0.5 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.001 --lambda_nvs 1.0  --back_culling \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/Synthetic4Relight/chair/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_syn4.py -m outputs/Synthetic4Relight/chair/$model   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_material_syn4.py -m outputs/Synthetic4Relight/chair/$model \
    --albedo_rescale 2   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_syn4.py -m outputs/Synthetic4Relight/chair/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light   --back_culling

# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/Synthetic4Relight/hotdog --eval \
    -m outputs/Synthetic4Relight/hotdog/$model  --iterations $iter \
    --start_checkpoint_refgs outputs/Synthetic4Relight/hotdog/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128  --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 1.0 --lambda_roughness_smooth 0.5 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.001 --lambda_nvs 1.0  --back_culling --light_t_min 0.1 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/Synthetic4Relight/hotdog/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_syn4.py -m outputs/Synthetic4Relight/hotdog/$model  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_material_syn4.py -m outputs/Synthetic4Relight/hotdog/$model --albedo_rescale 2  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_syn4.py -m outputs/Synthetic4Relight/hotdog/$model \
   --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light   --back_culling


# jugs
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/Synthetic4Relight/jugs --eval \
    -m outputs/Synthetic4Relight/jugs/$model --iterations $iter \
    --start_checkpoint_refgs outputs/Synthetic4Relight/jugs/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.8 \
    --lambda_base_color_smooth 1.0 --lambda_roughness_smooth 0.5 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.001 --lambda_nvs 1.0  --back_culling --light_t_min 0.1 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/Synthetic4Relight/jugs/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_syn4.py -m outputs/Synthetic4Relight/jugs/$model  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_material_syn4.py -m outputs/Synthetic4Relight/jugs/$model \
    --albedo_rescale 2  --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_syn4.py -m outputs/Synthetic4Relight/jugs/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light --back_culling