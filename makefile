PYTHON=python3

INPUT=trained_models/best.pt
OUTPUT_TRAIN=trained_models/
OUTPUT_EXPORT=output_models/model
EPOCHS=20


all: train_export clean

train_export:
	${PYTHON} -m webmnist --train -o ${OUTPUT_TRAIN} --epochs ${EPOCHS}
	${PYTHON} -m webmnist --export -i ${INPUT} -o ${OUTPUT_EXPORT}

train:
	${PYTHON} -m webmnist --train -o ${OUTPUT_TRAIN} --num-epochs ${EPOCHS}

export:
	${PYTHON} -m webmnist --export -i ${INPUT} -o ${OUTPUT_EXPORT}

clean:
	rm -rf ${OUTPUT}.pth ${OUTPUT}.onnx ${OUTPUT}.pb