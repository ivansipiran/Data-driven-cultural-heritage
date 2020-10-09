mkdir Results
cd Results
mkdir Airplane
mkdir Bag
mkdir Cap
mkdir Car
mkdir Chair
mkdir Guitar
mkdir Lamp
mkdir Laptop
mkdir Motorbike
mkdir Mug
mkdir Pistol
mkdir Skateboard
mkdir Table

cd ../..

python  shapenet_model_evaluation.py --model=pretrained/MBD_SHAPENET/ --outputFolder=Results/