from FER_directory.logic_ml.preprocessing_video import annotate_faces_in_video_light, annotate_faces_in_video_full

if __name__ == '__main__':

    input_video_path = 'test.mp4'
    output_video_path = 'output_video.mp4'
    model_path = 'models_trained/ResNet50V2_my_model_VM.h5'

    # annotate_faces_in_video_light(input_video_path, output_video_path, model_path, frame_skip=16)

    # #OR

    # annotate_faces_in_video_full(input_video_path, output_video_path, model_path)
