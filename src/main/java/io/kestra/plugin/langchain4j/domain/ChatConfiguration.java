package io.kestra.plugin.langchain4j.domain;

import io.kestra.core.models.property.Property;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ChatConfiguration {
    private Property<Double> temperature;
    private Property<Integer> topK;
    private Property<Double> topP;

    public static ChatConfiguration empty() {
        return ChatConfiguration.builder().build();
    }
}
