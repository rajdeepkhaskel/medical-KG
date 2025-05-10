### File structure
Please use the ```path_cache.py``` to perform the extraction and refining of node descriptions dynamically. Add the mistral API key. 
The hetionet-v1.0.json file needs to be unzipped.
The file ```extractor.py``` and ```refiner.py``` are used to extract and refine the node descriptions respectively and store them, to avoid dynamic extraction and refining of node descriptions during the PathRAG setup. The prompts can be improved a little for better outputs.