<span dir="rtl" align="right">

# סדנה: סיור ב - Hugging Face Hub 

<aside>

💡 **ברוכים הבאים**

הכנו ערכה שמדריכים באוניברסיטה יכולים להשתמש בה כדי להכין בקלות מעבדות, שיעורי בית, או מערכי שיעור. תוכן זה הוא **בחינם** ומשתמש בטכנולוגיות קוד פתוח מוכרות (`transformers`, `gradio`, וכו').

לחלופין, אפשר לבקש שמישהו מצוות Hugging Face יריץ את ההדרכות לכיתה שלכם דרך יוזמת ה - [ML demo.cratization tour](https://www.notion.so/ML-Demo-cratization-tour-with-66847a294abd4e9785e85663f5239652) !

אפשר למצוא את כל ההדרכות והמשאבים שהרכבנו [כאן](https://www.notion.so/Education-Toolkit-7b4a9a9d65ee4a6eb16178ec2a4f3599).

</aside>

**משך ההדרכה:** 20 עד 40 דקות

**מטרה:** ללמוד איך להשתמש ביעילות ב - [Hub platform](http://hf.co) החינמי ולהיות מסוגלים לשתף פעולה באקוסיסטם עם צוותים בפרויקטים של machine learning (ML).

מטרות למידה:

- לחקור את יותר מ 30,000 המודלים ששותפו ב - Hub.
- ללמוד דרכים יעילות למצוא את המודל ואת ה - datasets הנכונים.
- ללמוד איך לשתף ואיך לעבוד בשיתוף פעולה ב - ML workflow שלכם.
- לחקור ML demos שיצרו חברי הקהילה.

**פורמט:** מעבדה קצרה או תרגיל בית.

**קהל היעד:** סטודנטים מכל רקע שמעוניינים להשתמש במודלים קיימים או לשתף מודלים משלהם.

**דרישות קדם**

- הבנה ב - Machine Learning.
- (אופציונלי, אך מומלץ) ניסיון עם Git ([מדריך](https://learngitbranching.js.org/)).

## **למה להשתמש ב HUB?**

ה - HUB היא פלטפורמה מרכזית בה כל אחד יכול לחלוק ולחקור מודלים, datasets ו - demos. בעיית ה - "solve AI" לא תפתר על ידי חברה בודדת, אלא על ידי תרבות של שיתוף ידע ומשאבים. לכן, ה - ה Hub מתוכנן להיות אוסף המודלים, ה datasets וה demos הגדול ביותר בקוד פתוח.

הנה כמה עובדות על ה - Hugging Face Hub:

- נמצאים שם יותר מ 30,000 מודלים פומביים.
- יש שם מודלים לעיבוד שפה טבעית, לראיה ממוחשבת, שמע, וללמידה מחיזוקים!
- קיימים מודלים ליותר מ 180 שפות.
- כל ספריית ML יכולה להפיק ערך מה - Hub: מ - TensorFlow דרף PyTorch ועד אינטרגציות מתקדמות עם spaCy, SpeechBrain, ו - 20 ספריות נוספות.

## חקירת מודל

בואו נתחיל בהכרת המודלים. אפשר לגשת ליותר מ 30,000 ב [hf.co/models](http://hf.co/models). שם תראו את [gpt2](https://huggingface.co/gpt2) כאחד מהמודלים עם הכי הרבה הורדות. נלחץ עליו.

The website will take you to the model card when you click a model. A model card is a tool that documents models, providing helpful information about the models and being essential for discoverability and reproducibility.

The interface has many components, so let’s go through them:

[https://www.youtube.com/watch?v=XvSGPZFEjDY&feature=emb_imp_woyt](https://www.youtube.com/watch?v=XvSGPZFEjDY&feature=emb_imp_woyt)

- At the top, you can find different **tags** for things such as the task (*text generation, image classification*, etc.), frameworks (*PyTorch*, *TensorFlow*, etc.), the model’s language (*English*, *Arabic*, *etc.*), and license (*e.g. MIT*).

![](../../images/mode_card_tags.png)

- At the right column, you can play with the model directly in the browser using the *Inference API*. GPT2 is a text generation model, so it will generate additional text given an initial input. Try typing something like, “It was a bright and sunny day.”

![](../../images/model_card_inference_api.png)

- In the middle, you can go through the model card content. It has sections such as Intended uses & limitations, Training procedure, and Citation Info.

![](../../images/model_card_content.png)

Where does all this data come from? At Hugging Face, everything is based in **Git repositories** and is open-sourced. You can click the “Files and Versions” tab, which will allow you to see all the repository files, including the model weights. The model card is a markdown file **([README.md](http://README.md))** which on top of the content contains metadata such as the tags.

![](../../images/model_card_git.png)

Since all models are Git-based repositories, you get version control out of the box. Just as with GitHub, you can do things such as Git cloning, adding, committing, branching, and pushing. If you’ve never used Git before, we suggest the following [resource](https://learngitbranching.js.org/).

**Challenge 1**. Open the `config.json` file of the GPT2 repository. The config file contains hyperparameters as well as useful information for loading the model. From this file, answer:

- Which is the activation function?
- What is the vocabulary size?

## **Exploring Models**

So far, we’ve explored a single model. Let’s go wild! At the left of [https://huggingface.co/models](https://huggingface.co/models), you can filter for different things:

- **Tasks:** There is support for dozens of tasks in different domains: Computer Vision, Natural Language Processing, Audio, and more. You can click the +13 to see all available tasks.
  - **Libraries:** Although the Hub was originally for transformers models, the Hub has integration with dozens of libraries. You can find models of Keras, spaCy, allenNLP, and more.
- **Datasets:** The Hub also hosts thousands of datasets, as you’ll find more about later.

![](../../images/model_card_filters.png)

- **Languages:** Many of the models on the Hub are NLP-related. You can find models for hundreds of languages, including low-resource languages.

**Challenge 2**. How many token classification models are there in English?

**Challenge 3**. If you had to pick a Spanish model for Automatic Speech Recognition, which would you choose? (It can be any model for this task and language)

## Adding a model

Let’s say you want to upload a model to the Hub. This model could be a model of any ML library: Scikit-learn, Keras, Transformers, etc.

Let’s go through the steps:

1. Go to [huggingface.co/new](http://huggingface.co/new) to create a new model repository. The repositories you make can be either public or private.
2. You start with a public repo that has a model card. You can upload your model either by using the Web UI or by doing it with Git. If you’ve never used Git before, we suggest just using the Web interface. You can click Add File and drag and drop the files you want to add. If you want to understand the complete workflow, let’s go with the Git approach.

    1. Install both git and git-lfs installed on your system.
        1. Git: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
        2. Git-lfs: [https://git-lfs.github.com/](https://git-lfs.github.com/). Large files need to be uploaded with Git LFS.  Git does not work well once your files are above a few megabytes, which is frequent in ML. ML models can be up to gigabytes or terabytes! 🤯
    2. Clone the repository you just created

        ```python
        git clone https://huggingface.co/<your-username>/<your-model-id>
        ```

    3. Go to the directory and initialize Git LFS
        1. Optional. We already provide a list of common file extensions for the large files in `.gitattributes`, If the files you want to upload are not included in the `.gitattributes` file, you might need as shown here: You can do so with

            ```python
            git lfs track "*.your_extension"
            ```

            ```python
             git lfs install
            ```

    4. Add your files to the repository. The files depend on the framework/libraries you’re using. Overall, what is important is that you provide all artifacts required to load the model. For example:
        1. For TensorFlow, you might want to upload a SavedModel or `h5` file.
        2. For PyTorch, usually, it’s a `pytorch_model.bin`.
        3. For Scikit-Learn, it’s usually a `joblib` file.

        Here is an example in Python saving a Scikit-Learn model file.

        ```python
        from sklearn import linear_model
        reg = linear_model.LinearRegression()
        reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

        from joblib import dump, load
        dump(reg, 'model.joblib')
        ```

    5. Commit and push your files (make sure the saved file is within the repository)

    ```python
    git add .
    git commit -m "First model version"
    git push
    ```

And we're done! You can check your repository with all the recently added files!

![](../../images/model_card_updated_repo.png)

The UI allows you to explore the model files and commits and to see the diff introduced by each commit.

**Challenge 4**. It’s your turn! Upload a dummy model of the library of your choice.

Now that the model is in the Hub, others can find them! You can also collaborate with others easily by creating an organization. Hosting through the Hub allows a team to update repositories and do things you might be used to, such as working in branches and working collaboratively. The Hub also enables versioning in your models: if a model checkpoint is suddenly broken, you can always head back to a previous version.

At the top of the `README`, you can find some metadata. You will only find the license right now, but you can add more things. Let’s try some of it:

```yaml
 tags:
- es       # This will automatically be detected as a language tag.
- bert     # You can have additional tags for filtering
- text-classification # This will automatically be detected as a task tag.
datasets:
- llamas # This will link to a dataset on the Hub if it exists.
```

**Challenge 5**. Using the [documentation](https://huggingface.co/docs/hub/model-repos#how-are-model-tags-determined), change the default example in the widget.

The metadata allows people to discover your model quickly. Your model will now show up when you search for text classification models in Spanish. The model will also show up when looking at the dataset.

Wait...datasets?

## Datasets

With ML pipelines, you usually have a dataset to train the model. The Hub hosts around 3000 datasets that are open-sourced and free to use in multiple domains. On top of it, the open-source `datasets` [library](https://huggingface.co/docs/datasets/) allows the easy use of these datasets, including huge ones, using very convenient features such as streaming. This lab won't go through the library, but it does explain how to explore them.

Similar to models, you can head to [https://hf.co/datasets](https://hf.co/datasets). At the left, you can find different filters based on the task, license, and size of the dataset.

Let’s explore the [GLUE](https://huggingface.co/datasets/glue) dataset, which is a famous dataset used to test the performance of NLP models.

- Similar to model repositories, you have a dataset card that documents the dataset. If you scroll down a bit, you will find things such as the summary, the structure, and more.

![](../../images/datasets_card.png)

- At the top, you can explore a slice of the dataset directly in the browser. The GLUE dataset is divided into multiple sub-datasets (or subsets) that you can select, such as COLA and QNLI.

  ![](../../images/datasets_slices.png)

- At the right of the dataset card, you can see a list of models trained on this dataset.

![](../../images/datasets_models_trained.png)

**Challenge 6**. Search for the Common Voice dataset. Answer these questions:

- What tasks can the Common Voice dataset be used to?
- How many languages are covered in this dataset?
- Which are the dataset splits?

## ML Demos

Sharing your models and datasets is great, but creating an interactive, publicly available demo is even cooler. Demos of models are an increasingly important part of the ecosystem. Demos allow:

- model developers to easily **present** their work to a wide audience, such as in stakeholder presentations, conferences, and course projects
- to increase **reproducibility** in machine learning by lowering the barrier to test a model
- to share with a non-technical audience **the impact of a model**
- build a machine learning **portfolio**

There are Open-Source Python frameworks such as Gradio and Streamlit that allow building these demos very easily, and tools such as Hugging Face [Spaces](http://hf.co/spaces/launch) which allow to host and share them. As a follow-up lab, we recommend doing the **Build and Host Machine Learning Demos with Gradio & Hugging Face** tutorial.

> In this tutorial, you get to:
>
> - Explore ML demos created by the community.
> - Build a quick demo for your machine learning model in Python using the `gradio` library
> - Host the demos for free with Hugging Face Spaces
> - Add your demo to the Hugging Face org for your class or conference
>
> ***Duration: 20-40 minutes***
>
> 👉 [click here to access the tutorial](https://colab.research.google.com/github.com/huggingface/education-toolkit/tree/main/02_ml-demos-with-gradio.ipynb)
</span>