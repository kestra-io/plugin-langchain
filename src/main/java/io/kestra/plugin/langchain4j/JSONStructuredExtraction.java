package io.kestra.plugin.langchain4j;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
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
@Schema(
    title = "JSON Structured Extraction Task",
    description = "Generates JSON structured extraction using AI models (Ollama, OpenAI, Gemini)"
)
@Plugin(
    examples = {
        @io.kestra.core.models.annotations.Example(
            title = "JSON Structured Extraction using Ollama",
            full = true,
            code = {
                """
                id: json_structured_extraction_ollama
                namespace: company.team
                task:
                    id: json_structured_extraction
                    type: io.kestra.core.plugin.langchain4j.JSONStructuredExtraction
                    jsonFields:
                      - name
                      - City
                    schemaName: Person
                    prompt: Hello, my name is John, I live in Paris
                    provider:
                      type: OLLAMA
                      modelName: llama3
                      endpoint: http://localhost:11434
                """
            }
        ),
        @io.kestra.core.models.annotations.Example(
            title = "JSON Structured Extraction using OpenAI",
            full = true,
            code = {
                """
                id: json_structured_extraction_openai
                namespace: company.team
                task:
                    id: json_structured_extraction
                    type: io.kestra.core.plugin.langchain4j.JSONStructuredExtraction
                    jsonFields:
                      - name
                      - City
                    schemaName: Person
                    prompt: Hello, my name is John, I live in Paris
                    provider:
                      type: OPEN_AI
                      apiKey: your_openai_api_key
                      modelName: GPT-4
                """
            }
        )
    }
)
public class JSONStructuredExtraction extends Task implements RunnableTask<JSONStructuredExtraction.Output> {

    @Schema(title = "Text prompt", description = "The input prompt for the AI model.")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Schema Name", description = "The name of the JSON schema for structured extraction.")
    @NotNull
    private Property<String> schemaName;

    @Schema(title = "JSON Fields", description = "List of fields to extract from the text.")
    @NotNull
    private Property<List<String>> jsonFields;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public JSONStructuredExtraction.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        String renderedSchemaName = runContext.render(schemaName).as(String.class).orElseThrow();
        List<String> renderedJsonFields = Property.asList(jsonFields, runContext, String.class);

        // Get the appropriate model from the factory
        ChatLanguageModel model = this.provider.chatLanguageModel(runContext, configuration);

        // Build JSON schema
        ResponseFormat responseFormat = ResponseFormat.builder()
            .type(ResponseFormatType.JSON)
            .jsonSchema(JsonSchema.builder()
                .name(renderedSchemaName)
                .rootElement(buildDynamicSchema(renderedJsonFields))
                .build())
            .build();

        // Build request
        ChatRequest chatRequest = ChatRequest.builder()
            .responseFormat(responseFormat)
            .messages(UserMessage.from(renderedPrompt))
            .build();


        // Generate structured JSON output
        ChatResponse answer = model.chat(chatRequest);

        logger.debug("Generated Structured Extraction: {}", answer);

        return Output.builder()
            .schemaName(renderedSchemaName)
            .extractedJson(answer.aiMessage().text())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Schema Name", description = "The schema name used for the structured JSON extraction.")
        private final String schemaName;

        @Schema(title = "Extracted JSON", description = "The structured JSON output.")
        private final String extractedJson;
    }

    public static JsonObjectSchema buildDynamicSchema(List<String> fields) {
        JsonObjectSchema.Builder schemaBuilder = JsonObjectSchema.builder();
        fields.forEach(schemaBuilder::addStringProperty);
        schemaBuilder.required(fields);
        return schemaBuilder.build();
    }

}
