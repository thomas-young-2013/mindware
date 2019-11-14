# LFE
- Hyperparameters for LFE:
    - Num_bins: 200
    - Lower bound: -10
    - Upper bound: 10
    - Improvement threshold: 1%
    - Confidence threshold: 0.8
    - Cross-validation: 5-fold
    
- 6686 Meta-features are generated from 8 numerical datasets: yeast, vehicle, usps, semeion, pc4, mammography, 
magic_telescope, madelon. (Features in categorical datasets like fbis_wc and splice are always 'good' 
for all operations.)


- LFE is tested on 6 numerical datasets and 1 categorical dataset(Average F1 score of 15 runs):

Datasets | Before | After
-------|-------|-------|
Elevators(N)|0.8887|**0.8900**
Gisette(N)|0.9657|**0.9659**
Houses(N)|0.9779|**0.9786**
Wind(N)|0.8540|**0.8547**
Musk(N)|0.8999|**0.9007**
Eeg-eye-state(N)|**0.8795**|0.7515
Electricity(C)|**0.8656**|0.8219





- Conclusion:
    - Slight improvement on datasets with a large proportion of numerical features
    - Imbalanced meta-feature labels and poor final results if making predictions on categorical 
    features
    - 2 **significant** drawbacks:
        - **Offset on a numerical feature has no influence on the prediction of a transformation.** E.g. [-2,-1,0,1,2] and 
        [50,51,52,53,54] generate the same meta-feature. If *sigmoid* works well on [-2,-1,0,1,2], LFE will
        also apply *sigmoid* to [50,51,52,53,54]. This may lead to a decline on the final result. Maybe it's
        the reason why the performance on 'eeg-eye-state' drops since its numerical features are around 4000.
        - **Only work on binary classification.** Reason: The label of a meta-feature is generated if the performance of the feature on a binary 
        classification problem (one-vs-rest for multiclass problem) improves after a transformation. 
        Therefore, the test dataset should be a binary classification to ensure the same shape as MLP inputs.




 
