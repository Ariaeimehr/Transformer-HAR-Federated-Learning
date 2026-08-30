comms_round = 100
global_model = Transformer # وارد شده از اسکریپت جداگانه

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/drive/MyDrive/FedPer/weights.h5',
    save_weights_only=True,
    monitor='acc',
    mode='auto',
    save_best_only=False,
    verbose=1,
)

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_local_weight_list = list()
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)

    for client in client_names:
        local_model = Transformer
        local_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                            metrics=['accuracy'])
        
        # در روش FedPer گاهی تنها لایه‌های خاصی کپی می‌شوند
        local_model.set_weights(global_weights)

        local_model.fit(clients_batched[client], epochs=1, verbose=0, callbacks=[checkpoint_callback])

        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        K.clear_session()

    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights)
