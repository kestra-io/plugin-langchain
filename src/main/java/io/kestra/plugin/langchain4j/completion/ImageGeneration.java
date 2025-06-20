package io.kestra.plugin.langchain4j.completion;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.Response;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.domain.ModelProvider;
import io.kestra.plugin.langchain4j.domain.TokenUsage;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@Schema(
    title = "Generate an image with AI models.",
    description = "Generate images with a prompt using OpenAI's DALL-E 3 or Google Vertex AI."
)
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Generate an image using OpenAI (DALL-E 3).",
            full = true,
            code = {
                """
                id: image_generation
                namespace: company.team
                task:
                    id: image_generation
                    type: io.kestra.core.plugin.langchain4j.completion.ImageGeneration
                    prompt: "A futuristic cityscape at sunset"
                    provider:
                        type: io.kestra.plugin.langchain4j.provider.OpenAI
                        apiKey: "{{secret('OPENAI_API_KEY')}}"
                        modelName: dall-e-3
                        size: LARGE
                        download: false
                """
            }
        ),
    },
    beta = true,
    aliases = "io.kestra.plugin.langchain4j.ImageGeneration"
)

public class ImageGeneration extends Task implements RunnableTask<ImageGeneration.Output> {

    @Schema(title = "Image prompt", description = "The input prompt for the image generation model")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Language Model Provider")
    @NotNull
    @PluginProperty
    private ModelProvider provider;

    @Override
    public ImageGeneration.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();

        // Get the model
        ImageModel model = provider.imageModel(runContext);

        Response<Image> imageUrl = model.generate(renderedPrompt);
        logger.debug("Generated Image URL: {}", imageUrl.content().url());

        return Output.builder()
            .imageUrl(String.valueOf(imageUrl.content().url()))
            .tokenUsage(TokenUsage.from(imageUrl.tokenUsage()))
            .finishReason(imageUrl.finishReason())
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Generated image URL", description = "The URL of the generated image")
        private String imageUrl;

        @Schema(title = "Token usage")
        private TokenUsage tokenUsage;

        @Schema(title = "Finish reason")
        private FinishReason finishReason;
    }
}