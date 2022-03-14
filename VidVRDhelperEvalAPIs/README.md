
# https://github.com/xdshang/VidVRD-helper

This `VidVRDhelperEvalAPIs` is copyed from `VidVRD-helper/evaluation`, because we only use these API inside our evaluation scripts.

#### Prediction JSON Format for Visual Relation Detection
```json5
{
    "version": "VERSION 1.0",
    "results": {
        "3249280846": [                                 // video ID
            {                                           // a detected visual relation instance
                "triplet": ["dog", "bite", "frisbee"],  // relation triplet
                "score": 0.9,                           // confidence score
                "duration": [5, 8],                     // starting (inclusive) and ending (exclusive) frame IDs
                "sub_traj": [                           // subject trajectory
                    [9, 10, 45, 20],                    // bounding box at the starting frame
                    [10, 17, 46, 22],                   // [left, top, right, bottom] (all inclusive)
                    [10, 12, 40, 15]
                ],
                "obj_traj": [                           // object trajectory
                    [20, 10, 145, 150],
                    [60, 11, 195, 170],
                    [73, 12, 189, 148]
                ]
            },
            {                                           // another detected visual relation instance
                "triplet": ["adult", "watch", "dog"],
                "score": 0.68,
                "duration": [6, 8],
                "sub_traj": [ 
                    [89, 10, 128, 80], 
                    [90, 10, 128, 79]
                ],
                "obj_traj": [
                    [10, 17, 46, 22],
                    [10, 12, 40, 16]
                ]
            }
        ]
    },
    "external_data": {                                  // for challenge submission only
        "used": true,
        "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
    }
}
```

#### Prediction JSON Format for Visual Object Detection
Please refer to this [page](https://xdshang.github.io/docs/video-relation-challenge/mm19-gdc/task1.html)

#### Prediction JSON Format for Action Detection
Please refer to this [page](https://xdshang.github.io/docs/video-relation-challenge/mm19-gdc/task2.html)
