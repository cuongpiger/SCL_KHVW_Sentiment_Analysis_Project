# 1. Description
* Hello guys, this is my small application used to predict comments are positive or negative.
* I used `selenium` to crawl the entire data from Shopee Vietnam with over than 23.000 records. Each record consists mainly of the client's comment and his number of stars for the his comment.
* After training phase and several model evaluation techniques, the model was embedded into web using Django framework.
* The main field that my app concentrate on is fashion. So if you try to force the model predict on a wrong data (such as technology comments, food comments,...), The possibility is that the model can not predict accurately.
* So, if you are interested in my application, you can read more the documents inside the directory `Docs`. I write with such a clarity and precision for what techniques I used.

# 2. Installation
## 2.1. Option 1: Naive way
* Firstly, you need to create a new **conda environment** using `python 3.8.10` and then activate the new environment which you have just created.
  ```bash
  conda create --name sentiment python==3.8.10 pip
  conda activate sentiment
  ```
* OK, open your terminal at the same level with file `requirements.txt` to install some python packages.
  ```bash
  pip install -r requirements.txt
  ```
* And then, change your directory to `web` directory.
  ```bash
  cd web
  ```
  and now, you are standing at path `.../SCL_KHVW_Sentiment_Analysis_Project/web/`
* Finally, you can run the Django application by the followed command.
  ```bash
  python manage.py runserver
  ```
  Because this application uses TensorFlow, SkLearn and some other heavy packages. So for the first time you run this app, it will take several seconds to load model and initialize some configuration, after that moment, the app can work smoothly.
* OK, open your favorite browser and go to url [http://localhost:8000](http://localhost:8000/), the super gorgeous interface of my app will appear like this 😅 (sorry front-end developers).
  ![](./images/rm_00.png)

## 2.2. Option 2: Docker way
- Run the followed command to run the application.
  ```bash
  docker run --name sentiment-analysis -p 8000:8000 -d manhcuong8499/sentiment-analysis
  ```
- [Optional] Verify the application is running.
  ```bash
  docker ps
  docker logs sentiment-analysis
  ```
  - The below result is the output of console:
  > Watching for file changes with StatReloader
  > Performing system checks...
  > 
  > 2023-10-11 04:12:50.799856: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
  > 2023-10-11 04:12:50.799869: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
  > System check identified no issues (0 silenced).
  > October 11, 2023 - 04:12:51
  > Django version 3.1.4, using settings 'shopee_sentiment.settings'
  > Starting development server at http://0.0.0.0:8000/
  > Quit the server with CONTROL-C.

- Visit the [http://localhost:8000](http://localhost:8000) to see the application.

# 3. Demo and usage
* Realtime video demo with so cute host: [https://youtu.be/iQgCI3D1mdo](https://youtu.be/iQgCI3D1mdo)
* After you run the application successfully, enter your comment at the textarea component, then click the button **Send** and wait a bit to get the predicted result.
* Let's try with some positive comments.
  ![](./images/rm_01.png)
  * With this comment, the model predict exactly. You can see **predicted probability of positive value** is higher than **negative value**, so really this is a positive comment.
  * Some images below for other positive comments.
    ![](./images/rm_02.png)
    ![](./images/rm_03.png)
    ![](images/rm_04.png)

* OK, let's try with some negative comments.
  * Let leave only the **thumbs-down** emoji to see how the model will deal with.
    ![](./images/rm_05.png)
  * Some images below for other negative comments.
    ![](images/rm_06.png)
    ![](images/rm_07.png)
    ![](images/rm_08.png)
    ![](images/rm_09.png)

* OK, cool. Thank you so much. Let clone or download as zip my application and run on your machine. See you 😍.