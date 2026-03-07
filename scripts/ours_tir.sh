cuda=${1:-0}
init_model=init
init_iter=40000
model=radiogs
iter=20000

# armadillo
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/armadillo \
    -m outputs/TensoIR/armadillo/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_radiosity 0.1 \
    --lambda_normal_smooth 0.02 \

# ficus
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/ficus \
    -m outputs/TensoIR/ficus/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1 

# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/hotdog \
    -m outputs/TensoIR/hotdog/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --lambda_normal_smooth 0.02 \
    # --train_sh_vol \
    # --radiosity_from_iter 2000 \
    # --lambda_radiosity 0.1 

# lego
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR/lego \
    -m outputs/TensoIR/lego/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1 

# armadillo
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/armadillo --eval \
    -m outputs/TensoIR/armadillo/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR/armadillo/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0 --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR/armadillo/$model --eval --skip_train \
    --diffuse_sample_num 64 --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR/armadillo/$model  
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR/armadillo/$model \
    --albedo_rescale 2   --no_lpips --no_save
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/TensoIR/armadillo/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light  --back_culling

# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/hotdog --eval \
    -m outputs/TensoIR/hotdog/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR/hotdog/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \
    --light_t_min 0.1

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR/hotdog/$model --eval --skip_train \
    --diffuse_sample_num 64  --light_t_min 0.1 --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR/hotdog/$model   --light_t_min 0.1
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR/hotdog/$model \
    --albedo_rescale 4   --light_t_min 0.1 
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/TensoIR/hotdog/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light   --light_t_min 0.1

# lego
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/lego --eval \
    -m outputs/TensoIR/lego/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR/lego/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.8 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR/lego/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR/lego/$model
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR/lego/$model \
    --albedo_rescale 2 --no_lpips --no_save
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/TensoIR/lego/$model \
    --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 4 -e light_scale4   --back_culling --envmaps fireplace forest

# ficus
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR/ficus --eval \
    -m outputs/TensoIR/ficus/$model  --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR/ficus/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128  --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.002 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling  --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR/ficus/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR/ficus/$model
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR/ficus/$model --albedo_rescale 2 \
     --no_lpips
CUDA_VISIBLE_DEVICES=$cuda python eval_relighting_tensoir.py -m outputs/TensoIR/ficus/$model \
   --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2 -e light   --back_culling
    