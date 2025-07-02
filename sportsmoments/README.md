
# SportsMoments Dataset

We present a novel fine-grained dataset for the task of Video-to-Video Moment Retrieval (*Vid2VidMR*) which contains information about temporal segments pertaining to fine-grained actions and moments such as *cover drive*, *goal kick*, *ducking a bouncer* or *penalty*. 

We release all the Youtube Video IDs in the `vids.txt` file. For downloading the raw query and reference clips, please visit the following [link](https://drive.google.com/drive/u/1/folders/1o3g7PZp8Czp6HFkS97HcWsxrcRJLw3nO)

Our annotation files consist of 3 splits : `train`, `val` and `test`. You can find the `train` file in the drive link above. Each file has the following fields :
```
    "query_id": unique identifier for query video,  
    "reference_id": unique identifier for reference video, 
    "reference_segment_start": start time of the moment, 
    "reference_segment_end": end time of the moment,
    "action_class": integer representing the class id of the moment/action, 
    "reference_duration": duration of the reference video
```
We release the `id_class_mapping.json` for mapping the `action_class` to its corresponding class name. In addition to the annotation files we also provide the visual captions associated with each of the query videos [here](https://drive.google.com/drive/u/1/folders/1o3g7PZp8Czp6HFkS97HcWsxrcRJLw3nO), if one wishes to evaluate `MATR` on textual queries.

In order to train the model, bring the `train` and `val` CSV files in [JSON Line](https://jsonlines.org/) format, such that each row of the files can be loaded as a single `dict` in Python. Below is an example of the annotation:

```
{
    "qid": "iGGElV40TcU__5_1816", 
    "query": "As the bowler delivers his bowl, the batsman swings and hits the ball, causing it to bounce. The batsman then runs towards the boundary, attempting to score as many runs as possible.....", 
    "duration": 133.7, 
    "vid": "MbdWVX2jUhA__5_889", 
    "relevant_windows": [[60.3, 73.3]],
    "relevant_clip_ids": [30, 31, 32, 33, 34, 35], 
    "saliency_scores": [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]]
}
```
`qid` corresponds to a query video paired with a reference video `vid`. Both of which are formatted as follows `{youtube_id}_{class_id}_{idx}`.
`duration` is an integer indicating the duration of this video.
`relevant_windows` is the list of windows that localize the moment, each window has two numbers, one indicates the start time of the moment, another one indicates the end time. `relevant_clip_ids` is the list of ids to the segmented 2-second clips that fall into the moments specified by `relevant_windows`.

Since the objective of this work does not encompass highlight detection, `saliency_scores` are all labelled as `4`, where each sublist corresponds to a clip in `relevant_clip_ids`.

**Note** : In order to train the model for Vid2VidMR, you can skip the `"query"` column.
