# Outputs

The results of the inference pipeline will be saved in the output directory specified by the `--output_dir` argument. The output will include:
- A CSV file containing the results of the inference, including species predictions, order predictions, and bounding box coordinates.
- A directory containing the cropped images of the detected objects, if the `--save_crops` argument is specified.
- A directory containing the original images, if the `--remove_image` argument is not specified.

The CSV file will have the following columns:

| column name  | description  |
|--------------|--------------|
| image_path   | The raw image path |
| image_datetime | image date and time |
| bucket_name  | bucket name |
| analysis_datetime | time of inference |
| recording_session | date of recording session |
| image_bluriness | the variance of laplace for the entire image |
| crop_status  | crop name |
| crop_bluriness | the variance of laplace for the crop |
| crop_area    | area of the crop in pixels |
| cropped_image_path | path for the crop, if output (optional) |
| box_score    | confidence of box/localisation |
| box_label    | box label (0, 1) |
| x_min        | box coordinates, x min in px |
| y_min        | box coordinates, y min in px |
| x_max        | box coordinates, x max in px |
| y_max        | box coordinates,y max in px |
| class_name   | binary class (moth/non-moth) |
| class_confidence | binary confidence (range 0, 1) |
| order_name   | order name |
| order_confidence | order confidence |
| top_1_species | most likely species prediction |
| ... | ... |
| top_[n]_species * | nth most likely species prediction |
| top_1_confidence | most likely species prediction confidence |
| ... | ... |
| top_[n]_confidence * | nth most likely species prediction confidence |
| previous_image | the previous image key |
| best_match_crop | the most similar crop from the previous image |
| cnn_cost | crop embedding similarity |
| iou_cost | intersection of union cost |
| box_ratio_cost | box overlap/area cost |
| dist_ratio_cost | distance between boxes |
| total_cost | (sum of all costs, equally weighted) |
| image_path_basename | current image path |
| crop_id | current crop |
| track_id | tracking identifier |

\* There are n species predictions included, as set by the `--top_n_species` argument in `perform_inferences`. If `--top_n_species` is not specified, the default is 5 species predictions.
