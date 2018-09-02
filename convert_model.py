from retinanet.keras_retinanet.bin import convert_model

path = 'trained_retinanet_models/'


def convert(path, file, backbone):
    infpath = path + 'inf/'
    print('converting...')
    convert_model.main(["--backbone=" + backbone,
                        path + file,
                        infpath + file])
    print('finish!')


convert(path, '0729_final_resnet50_csv_50_noweights.h5', 'resnet50')
convert(path, '0802_final_resnet50_csv_50_noweights_without_very_small_craters.h5', 'resnet50')
convert(path, '0805_final_resnet50_csv_50_withweights_without_very_small_craters.h5', 'resnet50')
# convert(path, '0806_final_resnet50_csv_50_withweights_with_very_small_craters_modified_anchors.h5', 'resnet50')
# convert(path, '0807_final_resnet50_csv_49_withweights_new_dataset_new_anchors_withsmallcraters.h5', 'resnet50')
# convert(path, '0813_final_resnet101_csv_50_withweights_new_dataset_new_anchors_withsmallcraters.h5', 'resnet101')
# convert(path, '0819_mobilenet224_1.0_csv_56.h5', 'mobilenet224_1.0')
# convert(path, '0817_final_resnet50_csv_50_withweights_random_transform_without_rotation.h5', 'resnet50')
# convert(path, '0823_final_resnet152_csv_73_withweights_FINALSETTINGS.h5', 'resnet152')
# convert(path, '0826_resnet101_csv_62_withweights_FINALSETTINGS.h5', 'resnet101')
# convert(path, '0830_resnet50_csv_56_withweights_FINALSETTINGS_1248px.h5', 'resnet50')
# convert(path, '0901_resnet152_csv_24_withweights_FINALSETTINGS_832px.h5', 'resnet152')
