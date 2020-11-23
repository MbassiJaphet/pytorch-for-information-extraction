import yaml

field_coordinates_file = './field_coordinates.yaml'
with open(field_coordinates_file) as f:
    field_coordinates = yaml.load(f, Loader=yaml.FullLoader)

# print field coordinates
field_coordinates