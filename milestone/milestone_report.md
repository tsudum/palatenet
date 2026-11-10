PalateNet milestone

PalateNet is about predicting how many stars someone might give a restaurant. We want to see whether the connections between people and the restaurants they review can help. If two people have reviewed some of the same places, those connections might give us useful information about their preferences.

We started with a small example using 2,000 real Yelp restaurant reviews. We turned each person and restaurant into a node in a graph, with a review connecting them. That example had 1,970 users and 1,140 restaurants. We ran one mean-aggregation layer, which lets each node take in the average of its neighbors' feature vectors. Those vectors were random placeholders, so this first notebook only showed that the graph and message passing worked. It did not predict ratings.

The second notebook added a trained rating predictor. We used restaurant reviews from Boise, Idaho and kept the earliest review when the same person reviewed the same restaurant more than once. This gave us 64,370 review pairs. We read the large review file one line at a time instead of loading it all into memory. Everything ran locally on CPU, without a GPU or cloud service.

We split the reviews by date so the model would learn from the past and predict later ratings. The oldest 60% formed the graph. The next 20% supplied training examples. A later 10% helped us choose when to stop training, and the newest 10% was saved for testing. Reviews we were trying to predict stayed out of the graph in both directions, so the model could not see the target review through its connections. The graph stayed fixed throughout the experiment.

The model learned a vector for each user and restaurant, then used mean aggregation before predicting a rating from the two vectors. It learned from the training ratings, but the older ratings stored on the graph were not used in its messages. For comparison, we fitted a simple baseline by averaging the same 3,674 training ratings. That baseline predicted about 3.78 stars for every test review. Both methods were tested on exactly the same reviews.

Across three training runs with different random seeds, our model had an MAE of 1.161 stars, compared with 1.264 for the baseline. MAE means the average size of a prediction's mistake, so our model was off by about 1.16 stars on average. Its RMSE was 1.389, compared with 1.496 for the baseline. RMSE gives bigger mistakes more weight. These results mean our model reduced MAE by 8.1% and RMSE by 7.1%. All three runs beat the baseline, although the remaining error is still fairly large.

There is an important limit to this result. We only tested reviews where both the person and restaurant already appeared in the older graph. That left 794 of the 6,437 reviews in the test period, or 12.3%. We do not yet know how well the model would work for new users or restaurants. Beating one average rating also does not prove that message passing caused the improvement. A useful next comparison would be each restaurant's past average rating, along with a version of our model that does not use message passing. More data might help, but we would need to test that rather than assume it.

The tutorial can follow the same path as the notebooks. It can start by explaining how reviews become a graph, then walk through the small message-passing example. The rating notebook can show how to keep future reviews separate and compare predictions with a baseline. Later work can explore whether combining mean, maximum, and standard deviation gives better results than mean alone.

The data came from the Yelp Open Dataset: https://business.yelp.com/data/resources/open-dataset/
