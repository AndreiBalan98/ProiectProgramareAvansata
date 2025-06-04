import javafx.application.Application;
import javafx.stage.Stage;
import userinterface.DigitsApplication;

public class Main extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        DigitsApplication app = new DigitsApplication();
        app.start(primaryStage);
    }
}