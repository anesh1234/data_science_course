# A dictionary to map class names to index values
class_to_index = {'Implant' : 0,
                'Fillings' : 1,
                'Impacted Tooth' : 2,
                'Cavity' : 3
                }

# Reverse mapping dictionary
index_to_class = {v: k for k, v in class_to_index.items()}