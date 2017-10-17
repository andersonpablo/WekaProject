import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TesteIBk {
	public static void main(String[] args) throws Exception {
		// ------------------------------------------------------
		// (1) importa��o da base de dados de treinamento
		// ------------------------------------------------------
		DataSource source = new DataSource("weather.arff");
		Instances D = source.getDataSet();

		// 1.1 - espeficica��o do atributo classe
		if (D.classIndex() == -1) {
			D.setClassIndex(D.numAttributes() - 1);
		}
		// ------------------------------------------------------
		// (2) Constru��o do modelo classificador (treinamento)
		// ------------------------------------------------------
		IBk k3 = new IBk(3);
		k3.buildClassifier(D);

		// ------------------------------------------------------
		// (3) Classificando um novo exemplo
		// ------------------------------------------------------

		// 3.1 cria��o de uma nova inst�ncia
		Instance newInst = new DenseInstance(5);
		newInst.setDataset(D);
		newInst.setValue(0, "sunny");
		newInst.setValue(1, 75);
		newInst.setValue(2, 80);
		newInst.setValue(3, "TRUE");
		
		// 3.2 classifica��o de uma nova inst�ncia
		double pred = k3.classifyInstance(newInst);

		// 3.3 imprime o valor de pred
		System.out.println("Predi��o: " + pred);

		
		Attribute a = D.attribute(4);
		String predClass = a.value((int) pred);
		System.out.println("Predi��o: " + predClass);
		
	}
}