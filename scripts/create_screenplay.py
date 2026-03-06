from utils.text_utils import TextUtils


def main():
    text_utils = TextUtils()
    json_schema = [
        {
            "scene": 1,
            "description": "A man is looking at his mobile phone. In the middle of a street"
        }
    ]
    prompt = "I need a music video for a song. Create 10 scenes for the video. An executive is looking at his mobile phone. A computer virus covert him to a Zombie"
    screenplay = text_utils.generate_dict_list(
        prompt,
        json_schema
    )
    screenplay_path = "output/screenplay.json"
    text_utils.save_dict_to_file(screenplay, screenplay_path)

if __name__=="__main__":
    main()
