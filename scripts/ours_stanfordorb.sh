cuda=${1:-0}
init_model=init
init_iter=40000
model=radiogs
iter=20000

scenes="
        cactus_scene001 cactus_scene005 cactus_scene007 \
        car_scene002 car_scene004 car_scene006 \
        gnome_scene003 gnome_scene005 gnome_scene007 \
        teapot_scene001 teapot_scene002 teapot_scene006 \
        "

for scene in $scenes; do

    CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/stanford-orb/$scene \
        -m outputs/stanford-orb/$scene/$init_model --eval -w \
        --lambda_mask_entropy 0.05 \
        --volume_render_until_iter $init_iter \
        --iterations $init_iter \
        --lambda_dist 1000 \
        --lambda_normal_smooth 0.02 \
        --lambda_light 0.01 \
        --lambda_normal_smooth 0.02 \
        --train_sh_vol \
        --lambda_radiosity 0.1 \
        --radiosity_from_iter 2000

    CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/stanford-orb/$scene --eval \
        -m outputs/stanford-orb/$scene/$model --iterations $iter \
        --start_checkpoint_refgs outputs/stanford-orb/$scene/$init_model/chkpnt$init_iter.pth \
        --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
        --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.01 \
        --lr_scale 0.01 --lambda_nvs 1.0 --back_culling \
        --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
        --use_rad_rndview --detach_rad_global \

    CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/stanford-orb/$scene/$model --eval --skip_train \
        --diffuse_sample_num 64 --back_culling
    CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_stanford.py -m outputs/stanford-orb/$scene/$model   
    CUDA_VISIBLE_DEVICES=$cuda python eval_material_stanford.py -m outputs/stanford-orb/$scene/$model \
        --albedo_rescale 2

done
 
python parse_results.py --base_dir outputs/stanford-orb --exp $model