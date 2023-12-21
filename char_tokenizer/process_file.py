def process_file(input_file, output_file):
    try:
        characters = set()
        # Open the input file for reading
        with open('vocab.txt', 'r', encoding='utf-8') as file:
            characters.update(file.read())

        with open(input_file, 'r', encoding='utf-8') as file:
            # Read all characters from the file into a set
            characters.update(file.read())

        sorted_characters = sorted(characters)

        # Open the output file for writing
        with open(output_file, 'w', encoding='utf-8') as output_file:
            # Write each character from the set on a new line
            for char in sorted_characters:
                output_file.write(char + '\n')

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
input_filename = 'input.txt'  # Replace with your input file name
output_filename = 'vocab.txt'  # Replace with your desired output file name

process_file(input_filename, output_filename)