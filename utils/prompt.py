def read_prompt_file(file_path):
    """Function to read the content of a prompt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return None
          
def update_prompt_file(file_path, new_prompt):
    """Function to update a prompt file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_prompt)
        return True
    except Exception as e:
        return False

