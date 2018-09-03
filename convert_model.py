from retinanet.keras_retinanet.bin import convert_model

path = 'trained_retinanet_models/'


def convert(path, file, backbone):
    infpath = path + 'inf/'
    print('converting...')
    convert_model.main(["--backbone=" + backbone,
                        path + file,
                        infpath + file])
    print('finish!')



