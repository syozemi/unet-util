import yaml

f = open("../U-Net_Gsan/settings.yml", encoding='UTF-8')
settings = yaml.load(f)
print(settings)
