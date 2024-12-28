# Mental Health Analytics Dashboard

This repository contains a Python script for a **Mental Health Analytics Dashboard** designed to provide an interactive platform for mental health assessment and analysis. The dashboard leverages modern web technologies and AI to offer insights into mental health conditions based on user inputs.

## Overview

The `mentalhealth_ai.py` script uses Dash by Plotly to create a web application that:

- **Assesses mental health** using the Patient Health Questionnaire-9 (PHQ-9) for depression screening.
- **Analyzes physical health metrics** to provide a holistic view of the user's well-being.
- **Generates personalized reports** with recommendations, risk levels, and support resources.
- **Utilizes AI** for generating assessments and ensuring responsible AI practices through bias analysis.

## Features

- **PHQ-9 Assessment**: Users can answer a series of questions to gauge their depression severity.
- **Physical Metrics Input**: Users provide data on various physiological metrics like heart rate, sleep quality, etc.
- **Contextual Notes**: Users can add additional context or notes for a more personalized assessment.
- **AI-Driven Analysis**: Uses OpenAI's API to generate assessments based on user inputs.
- **Bias Analysis**: Ensures the assessment is free from potential biases related to age, gender, ethnicity, and socioeconomic status.
- **Visualizations**: Generates charts and graphs for mood trends, wellness metrics, and sleep analysis.
- **PDF Report Generation**: Creates a downloadable PDF report summarizing the assessment.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone [your-repo-url]
   cd [your-repo-directory]
   ```

2. **Install Dependencies**:
   ```bash
   pip install dash dash-bootstrap-components plotly pandas numpy openai fpdf
   ```

3. **Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Run the Application**:
   ```bash
   python mentalhealth_ai.py
   ```

   The application will start on `http://127.0.0.1:8050/`.

## Usage

- **Navigate to the Dashboard**: Open the URL provided in your web browser.
- **Complete the Assessment**: Fill out the PHQ-9 questions, enter physical metrics, and provide any additional context.
- **Generate Report**: Click on the "Generate Assessment" button to see your results and download the report.

## Key Components

- **Dash**: For creating the interactive web application.
- **OpenAI**: For generating AI-driven mental health assessments.
- **SQLite**: For storing assessment data locally.
- **Plotly**: For creating visualizations.
- **FPDF**: For generating PDF reports.

## Responsible AI

The application includes:

- **Bias Analysis**: Checks for potential biases in the assessment results.
- **Confidence Scoring**: Provides a confidence score for the assessment to indicate reliability.

## Limitations

- **Screening Tool**: This dashboard is for screening purposes only. It does not replace professional mental health evaluation.
- **Data Privacy**: Ensure compliance with data protection laws when handling user data.

## Future Enhancements

- **Integration with Wearables**: Real-time data from fitness trackers.
- **Longitudinal Analysis**: Track changes in mental health over time.
- **Multi-language Support**: Make the dashboard accessible to non-English speakers.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to Plotly for the Dash framework.
- OpenAI for providing the AI capabilities.
- The mental health community for their invaluable insights and feedback.
