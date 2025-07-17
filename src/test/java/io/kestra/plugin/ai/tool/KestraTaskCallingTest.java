package io.kestra.plugin.ai.tool;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.plugin.ai.completion.ChatCompletion;
import io.kestra.plugin.ai.provider.OpenAI;
import io.kestra.plugin.core.log.Log;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@KestraTest
class KestraTaskCallingTest {
    @Inject
    private RunContextFactory runContextFactory;

    @Test
    void chat() throws Exception {
        RunContext runContext = runContextFactory.of(Map.of(
            "apiKey", "demo",
            "modelName", "gpt-4o-mini",
            "baseUrl", "http://langchain4j.dev/demo/openai/v1"
        ));

        var chat = ChatCompletion.builder()
            .provider(OpenAI.builder()
                .type(OpenAI.class.getName())
                .apiKey(Property.ofExpression("{{ apiKey }}"))
                .modelName(Property.ofExpression("{{ modelName }}"))
                .baseUrl(Property.ofExpression("{{ baseUrl }}"))
                .build()
            )
            .tools(List.of(
                KestraTaskCalling.builder().tasks(
                    List.of(
                        Log.builder().id("log").type(Log.class.getName()).message("{{agent.message}}").build()
                    )
                ).build()
            ))
            .messages(Property.ofValue(
                List.of(
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.SYSTEM).content("You are an AI agent, please use the provided tool to fulfill the request.").build(),
                    ChatCompletion.ChatMessage.builder().type(ChatCompletion.ChatMessageType.USER).content("I want to log the following message: \"Hello World!\"").build()
                )))
            .build();

        var output = chat.run(runContext);
        assertThat(output.getAiResponse()).contains("success");
    }
}