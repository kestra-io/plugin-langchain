package io.kestra.plugin.langchain4j;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.image.ImageModel;
import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.plugin.langchain4j.domain.ChatConfiguration;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Getter
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, visible = true, property = "type", include = JsonTypeInfo.As.EXISTING_PROPERTY)
public abstract class ModelProvider {
    @NotNull
    @PluginProperty
    private String type;

    @NotNull
    private Property<String> modelName;

    public abstract ChatLanguageModel chatLanguageModel(RunContext runContext, ChatConfiguration configuration) throws IllegalVariableEvaluationException;

    public abstract ImageModel imageModel(RunContext runContext) throws IllegalVariableEvaluationException;

    public abstract EmbeddingModel embeddingModel(RunContext runContext) throws IllegalVariableEvaluationException;
}
