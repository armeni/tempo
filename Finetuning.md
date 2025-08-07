### To train the model on our dataset

1. Prepare `OUR_DATA` with the following structure and put it into `./data`:
    
        ├── train
        │   ├── time1
        │   │   └── images1.png
        │   ├── time2
        │   │   └── images2.png
        │   └── label
        │       └── label.png
        ├── val (the same with train)
        └── test(the same with train)
2. Create `OUR_DATACD_config.py` file in `/configs/_base_` (use `SYSUCD_config.py` as a template).
3. Insert the path of this file into `configs/cdlamba.py`'s `_base_` variable. Change the `exp_name = "OUR_DATACD_epoch{}/{}".format(epoch, net)`.
4. Create `OUR_DATA_dataset.py` file in `/rscd/datasets` (use `sysucd_dataset` as a template).
5. Download the pretrained weights from [here](https://drive.google.com/drive/folders/1BrZU0339JAFpKsQf4kdS0EpeeFgrBvBJ?usp=drive_link) and put it into `/rscd/models/backbones/review_pretrain`.
6. Create `work_dirs/OUR_DATACD_epochX/CDLamba`.
7. Run the training `python train.py -c configs/cdlamba.py`.

### Testing

1. Run the command:
```
python test.py \
-c configs/cdlamba.py \
--ckpt work_dirs/OUR_DATACD_epochX/CDLamba/version_0/ckpts/test/epoch=Y.ckpt \
--output_dir work_dirs/OUR_DATACD_epochX/CDLamba/version_0/ckpts/test \
```
2. If it is needed to test on the another dataset, just change `configs/cdlamba.py`'s `_base_` variable.

Some command to merge dataset from several folders into one:

```
  for dir in *256; 
  do   if [ "$dir" != "merged256" ]; 
  then  cp "$dir/label"/* merged256/label/ 2>/dev/null;     
  cp "$dir/time1"/* merged256/time1/ 2>/dev/null;     
  cp "$dir/time2"/* merged256/time2/ 2>/dev/null;   
  fi; done 
```

*Maybe it will be necessary to change training strategy:
- If we need to validate on the f1 score difference between 2 datasets during training, see `train_diff.py`.
- Also in that case we need to add 2nd dataset in `configs/cdlamba.py`'s `_base_` variable and change variables' names in `/configs/_base_/OUR_DATACD_config.py` to `dataset_config1`, and in the 2nd config file to `dataset_config2`.