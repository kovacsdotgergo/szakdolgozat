# Processing of audio signals with deep learning

## Running the code:

After opening esc_notebook.ipynb in Google Colab the cells
- install required Python packages
- set up the environment and clone the required Git repositories
- instantiate the deep learning models
- instantiate the deep learning models
    - with random split
    - with split based on folds (3 fold for training, 1 for validation and 1 for test)
- train the deep learning models
    - use the model for inference
    - try out different learning rates
    - train the model
        - test the model
        - visualize the training process
        - visualize the possible inputs of the model

---
# Audiojelek feldolgozása mélytanulással 

## A kód futtatása:

Az esc_notebook.pynb fájl megnyitása után Google Colabban a cellák futtatásával elvégezhető: 
- szükséges Python csomagok installálása
- Git repository-k klónozása és környezet előkészítése
- mélytanuló modellek példányosítása
- adathalmaz, tanító osztály példányosítása
    - véletlenszerű felosztással 80-10-10% arányban
    - 3 tanító, 1 validációs és 1 teszt folddal
- mélytanuló modellek kipróbálása
    - adathalmaz címkéinek becslése
    - különböző lépésközök kipróbálása
    - tanítás
        - teszt
        - tanítás folyamatának ábrázolása
        - bemenetek ábrázolása

Github link: \
https://github.com/kovacsdotgergo/szakdolgozat

Sources / Források:
- AST: https://github.com/YuanGongND/ast 
- ESC-50: https://github.com/karolpiczak/ESC-50