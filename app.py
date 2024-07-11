# extract labels and inputs from the music data set and store it in a json file

import os
import librosa

DATA_PATH = "genre_altered"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30 #SECONDS
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION 

def save_mfcc(data_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segment=5):
    # dictionaty to stare data
    data = {
        "mapping":[],
        "mfcc":[],
        "labels":[]
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segment)
    
    # loop through all genres
    for i,(root,dirname,filename) in enumerate(os.walk(data_path)):
        # ensure we are not in the root level
        if root is not data_path:
            # save the semantic label to mapping
            semantic_components = root.split("/") #genre_altered/blues => ["genre_altered","blues"]
            semantic_label = semantic_components[-1] #select the last item in the array
            data["mapping"].append(semantic_label)
            
            # process files for current genre
            for f in filename:
                # get file path to load audio 
                file_path = os.path.join(root,f)
                signal ,sr = librosa.load(file_path)
                
                # segment the signal for mfcc extraction
                for s in range(num_segment):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    # find mfcc
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                n_mfcc=n_mfcc,
                                                sr=sr,
                                                hop_length=hop_length,
                                                n_fft = n_fft)
                    
                    mfcc = mfcc.T
                    
                
                
        
