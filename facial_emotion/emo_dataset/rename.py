import os
import uuid

image_directory = "surprise"

for file_name in os.listdir(image_directory):
    
    unique_id = str(uuid.uuid4())

    new_file_name=unique_id+'.png'
    
    old_file_path = os.path.join(image_directory, file_name)
    new_file_path = os.path.join(image_directory, new_file_name)

    os.rename(old_file_path, new_file_path)


