import os
import pickle 
import json
from tqdm import tqdm

def cvt_pkl2xz(predict_relations,save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    results = predict_relations

    count = 0
    for video_name, res in tqdm(results.items()):
        # print(video_name)
        # print(res[0].keys())

        res_ = []
        for r in res:
            sub_traj = r["sub_traj"]
            # sub_traj = [[round(int(x)) for x in xx] for xx in sub_traj]

            obj_traj = r["obj_traj"]
            # obj_traj = [[round(int(x)) for x in xx] for xx in obj_traj]

            res_.append(
                {
                    "triplet":r["triplet"],
                    "duration":r["duration"],
                    "score":r["score"],
                    "sub_traj":sub_traj,
                    "obj_traj":obj_traj
                }
            )

        result_per_video = {video_name:res_}
        save_path = os.path.join(save_dir,video_name)
        with open(save_path,'w') as f:
            json.dump(result_per_video,f)

        os.system("xz -z {}".format(save_path))  # xz compression
        count += 1
        # if count > 4:
        #     break




    ## --------- zip compression -------
    print("finish cvt-result, start zip compression...")
    os.system("zip -r {}.zip {}".format(save_dir,save_dir))

    abs_path = os.path.abspath("{}.zip".format(save_dir))
    print("all finished, result saved at {}".format(abs_path))
    os.system("du -sh {}".format(abs_path))
