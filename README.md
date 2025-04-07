# Yelp Reviews Sentiment Analysis

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scala](https://img.shields.io/badge/Scala-2.13.x-blue.svg)](https://www.scala-lang.org/)

## Description

The Yelp Sentiment Analysis project is implemented using Scala, Apache Spark, SparkML, SparkNLP, Apache Cassandra, and S3. This project aims to collect user reviews from various businesses and train a Logistic Regression model that can analyze incoming text to determine if it is positive, negative, or neutral.

## Table of Contents

* [Installation](#installation)
* [Datasets](#datasets)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

## Installation

First, you need to have Docker and Docker Compose installed on your system, as they are required to run Apache Cassandra and MinIO. Then, edit the `.env` file according to your settings. You can then control the program's execution settings in the `Configs` package.

Typically, Scala projects use a build tool like sbt or Maven. Choose the appropriate instructions below:

## Datasets

Please download the following datasets and place them in the `src/main/resources` directory of your project:

* **Yelp Review Dataset:** [https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?resource=download&select=yelp_academic_dataset_review.json](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?resource=download&select=yelp_academic_dataset_review.json)
    * After downloading, ensure the file `yelp_academic_dataset_review.json` is located in `src/main/resources`.

* **VADER Lexicon:** [https://www.kaggle.com/datasets/nltkdata/vader-lexicon](https://www.kaggle.com/datasets/nltkdata/vader-lexicon)
    * You might need to navigate within this Kaggle dataset to find the actual lexicon file (it's often named something like `vader_lexicon.txt` or similar). Download this file and place it in `src/main/resources`.

Make sure both downloaded files are present in the `src/main/resources` folder for the project to access them.

### Using sbt

1.  Make sure you have sbt installed on your system. You can find installation instructions [here](https://www.scala-sbt.org/download.html).
2.  Clone the repository:
    ```bash
    git clone [repository URL]
    ```
3.  Navigate to the project directory:
    ```bash
    cd [project name]
    ```
4.  Run the `sbt` command in your terminal.
5.  To compile the project, run:
    ```bash
    sbt compile
    ```
6.  To run the project, run:
    ```bash
    sbt run
    ```
7.  To create a package (e.g., a JAR file), run:
    ```bash
    sbt package
    ```

### Using Maven

1.  Make sure you have Maven installed on your system. You can find installation instructions [here](https://maven.apache.org/download.cgi).
2.  Clone the repository:
    ```bash
    git clone [repository URL]
    ```
3.  Navigate to the project directory:
    ```bash
    cd [project name]
    ```
4.  To compile the project, run:
    ```bash
    mvn compile
    ```
5.  To run the project (you might need to configure the `pom.xml` for executable JARs), run:
    ```bash
    mvn exec:java -Dexec.mainClass="[Your Main Class]"
    ```
6.  To create a package (e.g., a JAR file), run:
    ```bash
    mvn package
    ```

**Note:** Replace `[repository URL]`, `[project name]`, and `[Your Main Class]` with your project's specific information. You might also need to adjust the build commands based on your project's configuration in `build.sbt` or `pom.xml`.

## Usage

In the `Configs` package, there are several configurable modes for running the program.

Modes:

* **Data preparation to train the model and save to Cassandra**
    * How to configure:
        ```scala
        val SyncCassandraWithDataset = true
        val OnlySyncCassandra = true
        ```
* **Using prepared data and training the model. You can enable or disable the option to save to S3.**
    * How to configure:
        ```scala
        val SyncCassandraWithDataset = false
        val OnlySyncCassandra = false
        val S3SaveModel = true
        val S3LoadModel = false
        ```
* **Using a trained and prepared model**
    * How to configure:
        ```scala
        val SyncCassandraWithDataset = false
        val OnlySyncCassandra = false
        val S3SaveModel = false
        val S3LoadModel = true
        ```

For more detailed usage instructions, please refer to the project's documentation or specific files within the `Configs` package.

## Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please follow these guidelines:

**Ways to Contribute:**

* **Report Bugs:** If you find a bug, please open a new issue on GitHub with a clear description of the problem and steps to reproduce it.
* **Suggest Features:** If you have an idea for a new feature or improvement, feel free to open an issue to discuss it.
* **Submit Code Changes:** If you'd like to contribute code, please follow the pull request process outlined below.

**Pull Request Process:**

1.  **Fork the Repository:** Fork this repository to your own GitHub account.
2.  **Create a Branch:** Create a new branch with a descriptive name for your changes (e.g., `feature/new-feature` or `bugfix/issue-123`).
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **Make Your Changes:** Implement your changes, ensuring that your code follows any existing style guidelines and includes necessary tests.
4.  **Commit Your Changes:** Commit your changes with clear and concise commit messages.
    ```bash
    git commit -m "Add your descriptive commit message"
    ```
5.  **Push to Your Fork:** Push your local branch to your forked repository on GitHub.
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Submit a Pull Request:** Go to the original repository on GitHub and click the "New pull request" button. Select your forked repository and the branch you created, and then provide a clear and detailed description of your changes in the pull request.

**Code Style and Guidelines:**

Please try to adhere to the existing code style and conventions used in this project. If there are specific contribution guidelines, you can find them in the `CONTRIBUTING.md` file.

We appreciate your contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
