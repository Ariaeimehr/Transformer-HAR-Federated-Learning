def evaluate_model(model, test_data):
    y_true, y_pred = [], []
    for x_test, y_test in test_data:
        predictions = model.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        y_true.extend(np.argmax(y_test, axis=1))
        y_pred.extend(predicted_labels)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1-Score: {f1_weighted:.4f}")
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return (precision_weighted, recall_weighted, f1_weighted), (precision_macro, recall_macro, f1_macro), conf_matrix

# ارزیابی نهایی مدل گلوبال
(weighted_scores, macro_scores, confusion) = evaluate_model(global_model, test_batched)
