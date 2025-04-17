package io.kestra.plugin.langchain4j;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.domain.ChatConfiguration;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import io.kestra.core.models.property.Property;


import java.util.Map;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "RAG using OpenAI",
            full = true,
            code = {
                """
                id: rag_example_openai
                namespace: company.team
                task:
                    id: rag_task
                    type: io.kestra.core.plugin.langchain4j.Rag
                    prompt: 'What is AI?'
                    provider:
                        type: OPENAI
                        apiKey: your_openai_api_key
                        modelName: gpt-3.5-turbo
                    context: |
                        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
                        Machine Learning (ML) is a subset of AI that focuses on the development of algorithms.
                """
            }
        ),
        @Example(
            title = "RAG using Ollama",
            full = true,
            code = {
                """
                id: rag_example_ollama
                namespace: company.team
                task:
                    id: rag_task
                    type: io.kestra.core.plugin.langchain4j.Rag
                    prompt: 'What is AI?'
                    provider:
                        type: OLLAMA
                        modelName: tinydolphin
                        endPoint: http://localhost:11434
                    context: |
                        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
                        Machine Learning (ML) is a subset of AI that focuses on the development of algorithms.
                """
            }
        ),
        @Example(
            title = "RAG using GOOGLE Gemini",
            full = true,
            code = {
                """
                id: rag_example_ollama
                namespace: company.team
                task:
                    id: rag_task
                    type: io.kestra.core.plugin.langchain4j.Rag
                    prompt: 'What is AI?'
                    provider:
                        type: GOOGLE_GEMINI
                        modelName: gemini-1.5-flash
                        apiKey: your-google_api_key
                    context: |
                        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
                        Machine Learning (ML) is a subset of AI that focuses on the development of algorithms.
                """
            }
        )
    }
)
public class Rag extends Task implements RunnableTask<Rag.Output> {

    @Schema(title = "Prompt text", description = "The question or input text for the RAG task")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Context", description = "The context or knowledge base to retrieve information from")
    @NotNull
    private Property<String> context;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @NotNull
    @PluginProperty
    @Builder.Default
    private ChatConfiguration configuration = ChatConfiguration.empty();

    @Override
    public Rag.Output run(RunContext runContext) throws Exception {
        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        String renderedContext = runContext.render(context).as(String.class).orElseThrow();

        ChatLanguageModel chatModel = provider.chatLanguageModel(runContext, configuration);

        // Create a prompt template for RAG
        PromptTemplate promptTemplate = PromptTemplate.from(
            "Answer the following question based on the context provided:\n\n" +
                "Context:\n{{context}}\n\n" +
                "Question: {{question}}"
        );

        // Generate the prompt with the rendered context and question
        Prompt ragPrompt = promptTemplate.apply(
            Map.of(
                "context", renderedContext,
                "question", renderedPrompt
            )

        );

        // Generate the response using the chat model
        String response = chatModel.chat(ragPrompt.text());

        return Output.builder()
            .response(response)
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Response", description = "The generated response from the RAG task")
        private final String response;
    }
}