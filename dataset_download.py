import os,zipfile,sys,json,requests
import getopt


def extract_zip(folder, model_name):
    file_name = os.path.abspath(folder +'/' + model_name + '.zip') # get full path of files
    zip_ref = zipfile.ZipFile(file_name) # create zipfile object
    os.mkdir(file_name[:-4])
    zip_ref.extractall(file_name[:-4]) # extract file to dir
    zip_ref.close() # close file
    os.remove(file_name) # del file


def download_collection(owner_name, collection_name, folder): 

    if os.path.lexists(folder) == True:
        print("Folder already exist, verify if dataset is downloaded")
        return

    if sys.version_info[0] < 3:
        raise Exception("Python 3 or greater is required. Try running `python3 download_collection.py`")

    if not owner_name:
        print('Error: missing `-o <owner_name>` option')
        return

    if not collection_name:
        print('Error: missing `-c <collection_name>` option')
        return

    os.mkdir(folder)

    print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

    page = 1
    count = 0

    # The Fuel server URL.
    base_url ='https://fuel.gazebosim.org/'

    # Fuel server version.
    fuel_version = '1.0'

    # Path to get the models in the collection
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

    # Path to download a single model in the collection
    download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)

    # Iterate over the pages
    while True:
        url = base_url + fuel_version + next_url

        # Get the contents of the current page.
        r = requests.get(url)

        if not r or not r.text:
            break

        # Convert to JSON
        models = json.loads(r.text)

        # Compute the next page's URL
        page = page + 1
        next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
    
        # Download each model 
        for model in models:
            count+=1
            model_name = model['name']
            print ('Downloading (%d) %s' % (count, model_name))
            download = requests.get(download_url+model_name+'.zip', stream=True)
            with open(os.path.join(folder,model_name+'.zip'), 'wb') as fd:
                for chunk in download.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)
                extract_zip(folder, model_name)
    print('Done.')
