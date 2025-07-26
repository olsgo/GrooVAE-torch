# 'TOONTRACK' IS JUST AN EXAMPLE, REPLACE IT WITH ACTUAL DATASET NAME


# For original humanize data
python humanize_preprocess.py

# For toontrack data
python humanize_preprocess.py --dataset toontrack_humanize --data-dir data/midi_data/toontrack

# For any other dataset
python humanize_preprocess.py --dataset abbey_road_humanize --data-dir data/midi_data/abbey_road

------

# 1. Generate info.csv for your dataset

python data/generate_infocsv.py --dataset toontrack

# 2. Process preserving human characteristics

python humanize_preprocess.py --dataset toontrack --data-dir data/midi_data/toontrack

# 3. Fine-tune with the processed data

python train_finetune.py --dataset toontrack
