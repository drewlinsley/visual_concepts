for category in $(ls ../../data/imgs/) 
do
    for concept in $(ls ../../data/imgs/"$category") 
    do
        ipython test_vgg19.py "$category" "$concept"
    done
done

