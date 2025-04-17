package io.kestra.plugin.langchain4j;

import dev.langchain4j.model.chat.ChatLanguageModel;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.domain.ChatConfiguration;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

import java.util.List;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(title = "Text Classification Task", description = "Classifies text using AI models")
@Plugin(
    examples = {
        @Example(
            title = "Text Classification using OpenAI",
            full = true,
            code = {
                """
                id: openai_text_classification
                namespace: company.team
                task:
                    id: text_classification
                    type: io.kestra.core.plugin.langchain4j.Classification
                    prompt: "Is 'This is a joke' a good joke?"
                    classes:
                      - true
                      - false
                    provider:
                        type: OPEN_AI
                        apiKey: your_openai_api_key
                        modelName: gpt-4o
                """
            }
        ),
        @Example(
            title = "Text Classification using Ollama",
            full = true,
            code = {
                """
                id: ollama_text_classification
                namespace: company.team
                task:
                    id: text_classification
                    type: io.kestra.core.plugin.langchain4j.Classification
                    prompt: "Is 'This is a joke' a good joke?"
                    classes:
                      - true
                      - false
                    provider:
                        type: OLLAMA
                        modelName: llama3
                        endpoint: http://localhost:11434
                """
            }
        ),
        @Example(
            title = "Text Classification using Gemini",
            full = true,
            code = {
                """
                id: gemini_text_classification
                namespace: company.team
                task:
                    id: text_classification
                    type: io.kestra.core.plugin.langchain4j.Classification
                    prompt: "Classify the sentiment of this sentence: 'I love this product!'"
                    classes:
                      - positive
                      - negative
                      - neutral
                    provider:
                        type: GOOGLE_GEMINI
                        apiKey: your_gemini_api_key
                        modelName: gemini-1.5-flash
                """
            }
        )
    }
)
public class Classification extends Task implements RunnableTask<Classification.Output> {

    @Schema(title = "Text prompt", description = "The input text to classify.")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Classification Options", description = "The list of possible classification categories.")
    @NotNull
    private Property<List<String>> classes;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public Classification.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        List<String> renderedClasses = runContext.render(classes).asList(String.class);

        // Get the appropriate model from the factory
        ChatLanguageModel model = this.provider.chatLanguageModel(runContext, configuration);

        String classificationPrompt = renderedPrompt +
            "\nRespond by only one of the following classes by typing just the exact class name: " + renderedClasses;

        // Perform text classification
        String classificationResponse = model.chat(classificationPrompt);
        logger.debug("Generated Classification: {}", classificationResponse);

        return Output.builder()
            .classification(classificationResponse)
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Classification Result", description = "The classified category of the input text.")
        private final String classification;
    }
}
