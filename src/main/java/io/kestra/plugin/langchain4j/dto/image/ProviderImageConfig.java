package io.kestra.plugin.langchain4j.dto.image;

import io.swagger.v3.oas.annotations.media.Schema;
import io.kestra.core.models.property.Property;
import jakarta.validation.constraints.NotNull;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Schema(title = "Provider Configuration", description = "Configuration settings for different providers")
public class ProviderImageConfig {

    @Schema(title = "Provider Type", description = "Choose between OPENAI or GOOGLE_VERTEX")
    @NotNull
    private ProviderImage type;

    @Schema(title = "API Key", description = "API key for the provider (if required)")
    @NotNull
    private Property<String> apiKey;

    @Schema(title = "Model Name", description = "Model name to use (e.g., dall-e-3, imagegeneration@005)")
    @NotNull
    private Property<String> modelName;

    @Schema(title = "Google Vertex Project ID", description = "Google Cloud project ID for Vertex AI")
    private Property<String> projectId;

    @Schema(title = "Google Vertex Location", description = "Google Cloud location for Vertex AI (e.g., us-central1)")
    private Property<String> location;

    @Schema(title = "Endpoint", description = "Endpoint for Model Ai")
    private Property<String> endpoint;

    @Schema(title = "Google Vertex Publisher", description = "Publisher for Vertex AI (e.g., google)")
    private Property<String> publisher;

    @Schema(
            title = "The size of the generated images."
    )
    @Builder.Default
    @NotNull
    private Property<Size> size = Property.of(Size.LARGE);

    @Schema(
            title = "Whether to download the generated image",
            description = "If enable, the generated image will be downloaded inside Kestra's internal storage. Else, the URL of the generated image will be available as task output."
    )
    @Builder.Default
    @NotNull
    private Property<Boolean> download = Property.of(Boolean.FALSE);
}