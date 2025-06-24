import json
from tqdm import tqdm
from openai import OpenAI
import base64

api_key = YOUR_OPENAI_API_KEY  # Replace with your actual OpenAI API key
round_idx = 1
path_images = YOUR_IMAGE_PATH  # Replace with the actual path to your images
path_out = YOUR_OUTPUT_PATH  # Replace with the desired output path for filtered images

client = OpenAI(
    api_key=api_key,
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

API_RATE = {"gpt-4o": {"input": 2.5/1000000, "output": 10/1000000}, "gpt-4o-mini": {"input": 0.15/1000000, "output": 0.6/1000000}}

PROMPT = '''
You are evaluating an image to decide whether it should be filtered out for data generation purposes. 
An ideal image should provide clear environmental context for robots, as these images will be used to generate a list of tasks that robots can perform based on the given situation.
Specifically, images should be filtered out if they meet any of the following criteria:
1) too blurry (slight blurriness is acceptable if objects remain identifiable, but excessively blurry images should be excluded.), 
2) too dark (some darkness is acceptable as long as objects can still be discerned. However, images that are too dark to identify objects should be filtered out.), 
3) too zoomed-in (images that are overly focused on a single detail (e.g., close-ups of flowers or a single individual) and lack broader environmental context should be excluded.),
4) too far-out (images taken from too far away, like mroe than 100 feet away, or those that primarily capture abstract landscapes, making it difficult to infer meaningful tasks specific for the environment, should be filtered out).

Please provide feedback for each criterion and the overall decision in JSON format as shown in the example below:
`{"blurry":"blurry/ok","darkness":"too dark/quite dark/slightly dark/ok", "zoomed-in": "too zoomed-in/somewhat zoomed-in/ok", "far-out":"too far-out/somewhat far-out/ok", "decision":"keep/filter"`}
'''


MODELNAME = "gpt-4o"

def run_one_image(image_path):
  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
    model=MODELNAME,
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": PROMPT,
          },
          {
            "type": "image_url",
            "image_url": {
              "url":  f"data:image/jpeg;base64,{base64_image}"
            },
          },
        ],
      }
    ],
  )
  cost = response.usage.prompt_tokens * API_RATE[MODELNAME]["input"] + response.usage.completion_tokens *API_RATE[MODELNAME]["output"]
  response_text = response.choices[0].message.content.lower()
  try:
    response_text = "{"+response_text.replace("\n","").split("{")[1].split("}")[0].strip()+"}"
    response_text = json.loads(response_text)
  except:
    print("error processing json")
    return None, cost
  return response_text, cost


with open(path_images, "r") as file:
    paths = [l.strip() for l in file.readlines()]

gpt4_response = {}
total_cost = 0
pbar = tqdm(paths)
for path in pbar:
   gpt4_response[path], _cost = run_one_image(path)
   total_cost += _cost
   pbar.set_description(f"${total_cost:.3f}")

path_to_label = {}
# mapping gpt output to final decision
for path, response in gpt4_response.items():
    if response is None:
        continue
    label = response['decision'] if response['zoomed-in'] == "ok" else "filter"
    path_to_label[path] = label
   
# save the ones labeld as "keep"
kept_paths = []
for path, label in path_to_label.items():
   if label == "keep":
      kept_paths.append(path)

kept_paths = sorted(kept_paths)
with open(path_out, "w") as file:
   file.write("\n".join(kept_paths))

print(f"Number of pathskept: {len(kept_paths)}/{len(paths)} ({len(kept_paths)/len(paths)*100:.2f}%)")