package io.kestra.plugin.langchain4j;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.models.tasks.Task;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.dto.image.ImageModelFactory;
import io.kestra.plugin.langchain4j.dto.image.ProviderImage;
import io.kestra.plugin.langchain4j.dto.image.ProviderImageConfig;
import io.kestra.plugin.langchain4j.dto.image.Size;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.slf4j.Logger;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Plugin(
    examples = {
        @Example(
            title = "Image Generation using OpenAI (DALL-E 3)",
            full = true,
            code = {
                """
                id: openai_image_generation
                namespace: company.team
                task:
                    id: image_generation
                    type: io.kestra.core.plugin.langchain4j.ImageGeneration
                    prompt: "A futuristic cityscape at sunset"
                    provider:
                        type: OPENAI
                        apiKey: your_openai_api_key
                        modelName: dall-e-3
                        size: LARGE
                        download: false
                """
            }
        ),
        @Example(
            title = "Image Generation using Google Vertex AI",
            full = true,
            code = {
                """
                id: google_vertex_image_generation
                namespace: company.team
                task:
                    id: image_generation
                    type: io.kestra.core.plugin.langchain4j.ImageGeneration
                    prompt: A realistic portrait of a medieval knight
                    provider:
                        type: GOOGLE_VERTEX
                        apiKey: your_google_api_key
                        modelName: imagegeneration@005
                        projectId: my-gcp-project
                        location: us-central1
                        endpoint: us-central1-aiplatform.googleapis.com
                        publisher: google
                """
            }
        )
    }
)

public class ImageGeneration extends Task implements RunnableTask<ImageGeneration.Output> {

    @Schema(title = "Image prompt", description = "The input prompt for the image generation model")
    @NotNull
    private Property<String> prompt;

    @Schema(title = "Provider Configuration", description = "Configuration for the provider (e.g., API key, model name)")
    @NotNull
    private ProviderImageConfig provider;


    @Override
    public ImageGeneration.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        // Render input properties
        String renderedPrompt = runContext.render(prompt).as(String.class).orElseThrow();
        ProviderImage renderedProviderType = provider.getType();
        String renderedModelName = runContext.render(provider.getModelName()).as(String.class).orElse(null);
        String renderedApiKey = runContext.render(provider.getApiKey()).as(String.class).orElse(null);
        String renderedProjectId = runContext.render(provider.getProjectId()).as(String.class).orElse(null);
        String renderedLocation = runContext.render(provider.getLocation()).as(String.class).orElse(null);
        String renderedEndpoint = runContext.render(provider.getEndpoint()).as(String.class).orElse(null);
        String renderedPublisher = runContext.render(provider.getPublisher()).as(String.class).orElse(null);
        Boolean renderedDownload = runContext.render(provider.getDownload()).as(Boolean.class).orElse(false);
        String renderedSize = runContext.render(provider.getSize()).as(Size.class).orElseThrow().getSize();

        // Get the model
        ImageModel model = ImageModelFactory.createModel(
            renderedProviderType,
            renderedApiKey,
            renderedModelName,
            renderedProjectId,
            renderedLocation,
            renderedEndpoint,
            renderedPublisher,
            renderedSize,
            renderedDownload
        );

        Response<Image> imageUrl = model.generate(renderedPrompt);
        logger.info("Generated Image URL: {}", imageUrl.content().url());

        return Output.builder()
            .imageUrl(String.valueOf(imageUrl.content().url()))
            .build();
    }
    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Generated image URL", description = "The URL of the generated image")
        private final String imageUrl;
    }
}