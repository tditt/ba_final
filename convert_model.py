from retinanet.keras_retinanet.bin import convert_model
convert_model.main(["--backbone=resnet152", 'trained_retinanet_models/final_resnet152_csv_73_withweights_FINALSETTINGS.h5', "inf_experimental/final_resnet152_csv_73_withweights_FINALSETTINGS.h5"])
