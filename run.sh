#/usr/bin/bash

# Clean up the directories so we are starting from scratch 
## TODO 
#if [ -f "$FILE" ]; then
#    echo "$FILE exist"
#fi

#if [ -f expanded_pratik_TEST.pkl ]; then 
#    rm expanded_pratik_TEST.pkl
#fi
echo "Clearning out the directories..." 
rm data/expanded_pratik_TEST.pkl 
rm data/expanded_shalaka_logical_firefox_TEST.pkl
rm data/expanded_pratik_TRAIN.pkl
rm data/expanded_shalaka_logical_firefox_TRAIN.pkl
rm outputs/expanded_pratik_TEST.csv
rm outputs/expanded_pratik_TEST.pdf
rm outputs/expanded_pratik_PCA.pdf
rm outputs/expanded_shalaka_logical_firefox_TEST.csv
rm outputs/expanded_shalaka_logical_firefox_TEST.pdf
rm outputs/expanded_shalaka_logical_firefox_PCA.pdf

# SHALAKA 
echo "Running Shalaka's data..."
python preprocess.py -i data/expanded_shalaka_logical_firefox.csv -f 5 -p 0.95
python train.py -i data/expanded_shalaka_logical_firefox_TRAIN.pkl -m models/mlaba_shalaka.pkl -v -s 
python test.py -m models/mlaba_shalaka.pkl -i data/expanded_shalaka_logical_firefox_TEST.pkl -o outputs/expanded_shalaka_logical_firefox_TEST


# PRATIK  
echo "Running Pratik's data..."
python preprocess.py -i data/expanded_pratik.csv -f 5 -p 0.95
python train.py -i data/expanded_pratik_TRAIN.pkl -m models/mlaba_pratik.pkl -v -s
python test.py -m models/mlaba_shalaka.pkl -i data/expanded_pratik_TEST.pkl -o outputs/expanded_pratik_TEST