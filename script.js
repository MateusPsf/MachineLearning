async function preverSalario() {

    // Pegando elementos da tela
    const textoStatus = document.getElementById("status");
    const textoResultado = document.getElementById("resultado");

    // Pegando valor digitado pelo usuário
    const anosExperiencia = Number(document.getElementById("anos").value);

    textoStatus.innerText = "Status: Treinando a IA...";

    // =========================
    // 1. CRIAR O MODELO
    // =========================
    const modelo = tf.sequential();
    modelo.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    // =========================
    // 2. COMPILAR O MODELO
    // =========================
    modelo.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    // =========================
    // 3. DADOS DE TREINO
    // X = anos de experiência
    // Y = salário
    // =========================
    const dadosEntrada = tf.tensor2d([1, 2, 3, 5, 7, 10], [6, 1]);
    const dadosSaida = tf.tensor2d([1500, 2500, 3500, 5000, 7000, 10000], [6, 1]);

    // =========================
    // 4. TREINAMENTO
    // =========================
    await modelo.fit(dadosEntrada, dadosSaida, {
        epochs: 200
    });

    textoStatus.innerText = "Status: IA treinada!";

    // =========================
    // 5. PREVISÃO
    // =========================
    const previsao = modelo.predict(
        tf.tensor2d([anosExperiencia], [1, 1])
    );

    const valorPrevisto = previsao.dataSync()[0];

    // =========================
    // 6. RESULTADO
    // =========================
    textoResultado.innerText =
        "Salário Previsto: R$ " + valorPrevisto.toFixed(2);
}