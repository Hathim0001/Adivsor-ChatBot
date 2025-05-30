# Automated Contract Clause Understanding and Risk Assessment with fine-tuned Legal-BERT and GPT-4o


This project aims to automate the understanding and risk assessment of contract clauses using a fine-tuned Legal-BERT model for classification and GPT-4 for generating detailed risk analysis and integrated explanations.

## Use Case and Importance

Legal professionals often need to review and assess contract clauses for potential risks and implications. This project provides an automated solution to classify clauses (e.g., identifying audit clauses) and analyze potential risks, streamlining the review process and enhancing productivity. This tool can be particularly valuable for lawyers, legal analysts, and compliance officers.

**Audit Clause**:
An audit clause is a provision in a contract that grants one party the right to inspect and review the records and operations of the other party to ensure compliance with the terms of the contract. It is crucial for maintaining transparency and accountability, especially in business agreements where financial and operational integrity is essential.

**Importance**:
Transparency: Ensures both parties are adhering to the contract terms.
Risk Management: Helps in identifying any discrepancies or potential risks early.
Compliance: Ensures legal and regulatory compliance.

## Technologies Used

- **Python**: The core programming language used.
- **Hugging Face Transformers**: For loading and fine-tuning the Legal-BERT model.
- **OpenAI GPT-4**: For generating risk analysis and integrated explanations.
- **Streamlit**: For creating the interactive web application.
- **Streamlit-Chat**: For implementing the chat interface.

## Model Training

The Legal-BERT model was fine-tuned using the `nlpaueb/legal-bert-base-uncased` checkpoint. The dataset was split into 5 folds for cross-validation to ensure robustness and to mitigate overfitting given the small dataset size.

## Results

### Evaluation Metrics

- **Accuracy**: Average of 99.23%
- **Precision**: Average of 98.49%
- **Recall**: Average of 100%
- **F1 Score**: Average of 99.24%

## Demo:
   - Enter a contract clause in the chat input.
   - The model will classify the clause and provide a risk assessment and integrated explanation.

https://github.com/Prakarsha01/Legal-Advice-chatbot/assets/67196711/f00072ba-dbcd-483d-8135-f0600eeba3b4

For detailed explaination on the project read the following [article](https://medium.com/@prakarsha/automated-contract-clause-understanding-and-risk-assessment-with-fine-tuned-legal-bert-and-gpt-4o-3a6f0423ace3).


## Conclusion

This project demonstrates the integration of a fine-tuned Legal-BERT model and GPT-4 to automate the classification and risk assessment of contract clauses. The results indicate high accuracy, making this tool potentially valuable for legal professionals seeking to streamline their contract review processes.

## References

- [Medium article: Detailed Project Explaination](https://medium.com/@prakarsha/automated-contract-clause-understanding-and-risk-assessment-with-fine-tuned-legal-bert-and-gpt-4o-3a6f0423ace3)
- [LegalBench Dataset](https://huggingface.co/datasets/nguha/legalbench)
- [LegalBench Research Paper](https://arxiv.org/abs/2308.11462)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI API](https://openai.com/api/)
- [Streamlit](https://streamlit.io/)
- [Cross Validation Techniques](https://machinelearningmastery.com/k-fold-cross-validation/)
- [Fine Tuned Legal-Bert](https://huggingface.co/Prakarsha01/fine-tuned-legal-bert-5folds)


Feel free to explore the code and contribute to improving the model and its applications. 

---
